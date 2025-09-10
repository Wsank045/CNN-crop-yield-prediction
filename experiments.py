from dataset import prepare_datasets_with_normalization, load_and_split, load_cluster_data
from model import CNNRegressor
from train import train_model, train_model_transfer_learning
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config(config: Dict) -> None:
    """Validate the experiment configuration dictionary"""
    required_keys = [
        'batch_size', 'num_workers', 'num_epochs', 
        'learning_rate', 'input_channels', 'dropout_rate',
        'results_dir', 'finetune_epochs', 'finetune_lr',
        'num_clusters'
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: Dict,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders with proper settings"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle_train,
        num_workers=config['num_workers'],
        persistent_workers=config['num_workers'] > 0 and torch.cuda.is_available(),
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        persistent_workers=config['num_workers'] > 0 and torch.cuda.is_available(),
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, test_loader

def save_results_plot(
    results: Dict,
    title: str,
    log_path: str,
    field_name: str = ""
) -> None:
    """Save and display results visualization plot"""
    plt.figure(figsize=(10, 8))
    plt.scatter(results['actuals'], results['predictions'], alpha=0.6, s=50)
    plt.plot([results['actuals'].min(), results['actuals'].max()],
             [results['actuals'].min(), results['actuals'].max()], 'r--', lw=2)
    plt.xlabel('Actual Yield', fontsize=12)
    plt.ylabel('Predicted Yield', fontsize=12)
    plt.title(f'{title} {field_name} Results (R² = {results["r2"]:.3f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plot_path = log_path.replace('.txt', '_predictions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  

def run_baseline_experiment(field_paths: Dict[str, str], config: Optional[Dict] = None) -> None:
    """Run baseline experiments for each field independently"""
    if config is None:
        raise ValueError('Config cannot be None')
    validate_config(config)
    
    logger.info("\n%s", '='*60)
    logger.info("RUNNING EXPERIMENT: BASELINE")

    for field_name, file_path in field_paths.items():
        try:
            logger.info("Loading %s from %s", field_name, file_path)
            df = pd.read_pickle(file_path)
            logger.info("Loaded %d samples", len(df))
            df['field_name'] = field_name

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logger.info("Train set: %d samples, Test set: %d samples", len(train_df), len(test_df))

            train_dataset, test_dataset, norm_config = prepare_datasets_with_normalization(
                train_df, test_df, config)

            train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, config)

            model = CNNRegressor(
                input_channels=config['input_channels'],
                dropout_rate=config['dropout_rate']
            )

            log_dir = Path(config['results_dir']) / 'baseline'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(log_dir / f"{field_name}.txt")

            final_results = train_model(
                model, train_loader, test_loader, 
                epochs=config['num_epochs'],
                lr=config['learning_rate'],
                norm_config=norm_config,
                log_path=log_path
            )

            final_results['field'] = field_name
            final_results['strategy'] = 'baseline'

            save_results_plot(
                final_results,
                "Baseline",
                log_path,
                field_name
            )

        except Exception as e:
            logger.error("Error in baseline experiment for %s: %s", field_name, e, exc_info=True)
            continue

def run_mixed_experiment(field_paths: Dict[str, str], config: Optional[Dict] = None) -> Optional[Dict]:
    """Run mixed experiment combining data from all fields"""
    if config is None:
        raise ValueError('Config cannot be None')
    validate_config(config)
    
    logger.info("\n%s", '='*60)
    logger.info("RUNNING EXPERIMENT: MIXED")

    try:
        train_df, test_df = load_and_split(field_paths=field_paths, strategy='mixed')
        logger.info("\n Data split complete")
        logger.info("Training: %d samples", len(train_df))
        logger.info("Testing: %d samples", len(test_df))

        train_dataset, test_dataset, norm_config = prepare_datasets_with_normalization(
            train_df, test_df, config
        )

        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, config)

        model = CNNRegressor(
            input_channels=config['input_channels'],
            dropout_rate=config['dropout_rate']
        )

        name = '_'.join(field_paths.keys())
        log_dir = Path(config['results_dir']) / 'mixed'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"{name}.txt")

        final_results = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config['num_epochs'],
            lr=config['learning_rate'],
            norm_config=norm_config,
            log_path=log_path,
        )

        save_results_plot(final_results, "MIXED", log_path)
        final_results['strategy'] = 'mixed'

        return final_results

    except Exception as e:
        logger.error("Error in mixed experiment: %s", e, exc_info=True)
        return None

def run_lofo_experiment(
    field_paths: Dict[str, str], 
    target_field: str, 
    config: Optional[Dict] = None
) -> Optional[Dict]:
    
    """Run Leave-One-Field-Out experiment"""
    if config is None:
        raise ValueError('Config cannot be None')
    validate_config(config)
    
    logger.info("\n%s", '='*60)
    logger.info("RUNNING EXPERIMENT: LOFO")

    try:
        train_df, test_df = load_and_split(
            field_paths=field_paths, 
            strategy='lofo',
            target_field=target_field
        )
        logger.info("\nData split complete")
        logger.info("Training: %d samples", len(train_df))
        logger.info("Testing: %d samples", len(test_df))

        train_dataset, test_dataset, norm_config = prepare_datasets_with_normalization(
            train_df, test_df, config
        )

        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, config)

        model = CNNRegressor(
            input_channels=config['input_channels'],
            dropout_rate=config['dropout_rate']
        )

        source_fields = [f for f in field_paths.keys() if f != target_field]
        log_dir = Path(config['results_dir']) / 'lofo'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"{'_'.join(source_fields)}_to_{target_field}.txt")

        final_results = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config['num_epochs'],
            lr=config['learning_rate'],
            norm_config=norm_config,
            log_path=log_path,
        )

        save_results_plot(final_results, "LOFO", log_path)
        final_results['strategy'] = 'lofo'

        return final_results

    except Exception as e:
        logger.error("Error in lofo experiment: %s", e, exc_info=True)
        return None

def run_tl_experiment(
    field_paths: Dict[str, str],
    target_field: str,
    config: Optional[Dict] = None,
    finetune_size: float = 0.3
) -> Optional[Dict]:
    """Run Transfer Learning experiment"""
    if config is None:
        raise ValueError('Config cannot be None')
    validate_config(config)
    
    logger.info("\n%s", '='*60)
    logger.info("RUNNING EXPERIMENT: TRANSFER LEARNING")

    try:
        finetune_df, test_df, retest_df = load_and_split(
            field_paths=field_paths,
            strategy='transfer_learning',
            target_field=target_field,
            fine_tune_size=finetune_size
        )

        logger.info("\n📊 Data prepared for transfer learning")
        logger.info("Fine-tuning: %d samples", len(finetune_df))
        logger.info("Testing: %d samples", len(test_df))
        logger.info("Re-testing: %d samples", len(retest_df))

        finetune_dataset, test_dataset, norm_config = prepare_datasets_with_normalization(
            finetune_df, test_df, config
        )
        _, retest_dataset, _ = prepare_datasets_with_normalization(
            finetune_df, retest_df, config
        )

        finetune_loader, _ = create_dataloaders(finetune_dataset, None, config, shuffle_train=True)
        test_loader, _ = create_dataloaders(test_dataset, None, config, shuffle_train=False)
        retest_loader, _ = create_dataloaders(retest_dataset, None, config, shuffle_train=False)

        # Determine model path
        source_fields = [f for f in field_paths.keys() if f != target_field]
        if len(source_fields) > 1:
            model_dir = Path(config['results_dir']) / 'mixed'
            model_name = '_'.join(source_fields) + '.best_model.pth'
        else:
            model_dir = Path(config['results_dir']) / 'baseline'
            model_name = source_fields[0] + '.best_model.pth'
        
        model_path = str(model_dir / model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained model not found at {model_path}")

        log_dir = Path(config['results_dir']) / 'tl'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"{'_'.join(source_fields)}_to_{target_field}_{finetune_size}.txt")

        final_results, _ = train_model_transfer_learning(
            model_path=model_path,
            finetune_loader=finetune_loader,
            test_loader=test_loader,
            training_test_loader=retest_loader,
            finetune_epochs=config['finetune_epochs'],
            finetune_lr=config['finetune_lr'],
            log_path=log_path,
            norm_config=norm_config,
            config=config
        )

        save_results_plot(final_results, "TL", log_path)
        final_results['strategy'] = 'transfer_learning'
        final_results['finetune_size'] = finetune_size

        return final_results

    except Exception as e:
        logger.error("Error in transfer learning experiment: %s", e, exc_info=True)
        return None

def run_cluster_experiment(
    field_path: str,
    target_field: str,
    config: Optional[Dict] = None,
    rev_cluster: bool = False,
    cluster_id: Optional[int] = None
) -> Optional[Dict]:
    """Run cluster-based experiment"""
    if config is None:
        raise ValueError('Config cannot be None')
    if cluster_id is None:
        raise ValueError('cluster_id must be specified')
    validate_config(config)
    
    logger.info("\nEXPERIMENT: CLUSTER RESULTS")
    logger.info("Running cluster experiment for field: %s", target_field)
    logger.info("\n%s", '='*60)

    try:
        train_df, test_df = load_cluster_data(
            field_path=field_path,
            target_field=target_field,
            num_clusters=config['num_clusters'],
            rev_cluster=rev_cluster,
            cluster_id=cluster_id
        )

        logger.info("Data split complete (combined clusters)")
        logger.info("Training samples: %d", len(train_df))
        logger.info("Testing samples: %d", len(test_df))

        train_dataset, test_dataset, norm_config = prepare_datasets_with_normalization(
            train_df, test_df, config
        )

        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, config)

        model = CNNRegressor(
            input_channels=config['input_channels'],
            dropout_rate=config['dropout_rate']
        )

        cluster_type = 'rev_cluster' if rev_cluster else 'cluster'
        log_dir = Path(config['results_dir']) / cluster_type
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"{target_field}_cluster_{cluster_id}.txt")

        final_results = train_model(
            model, 
            train_loader, 
            test_loader,
            epochs=config['num_epochs'],
            lr=config['learning_rate'],
            log_path=log_path,
            norm_config=norm_config
        )
        
        final_results['field'] = target_field

        save_results_plot(final_results, "Clusters", log_path, field_name=target_field)
        final_results['strategy'] = cluster_type
        final_results['cluster_id'] = cluster_id

        return final_results

    except Exception as e:
        logger.error("Error in cluster experiment: %s", e, exc_info=True)
        return None