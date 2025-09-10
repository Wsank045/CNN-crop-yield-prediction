import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from datetime import datetime
import torch.optim as optim
import os
import math
import traceback
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
from model import CNNRegressor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class EarlyStopping:
    """EarlyStopping"""
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.restore_best_weights = restore_best_weights
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: Optional[nn.Module] = None) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None and self.restore_best_weights:
                self.best_state_dict = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_state_dict = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")

    def restore_model(self, model: nn.Module) -> None:
        """Restore model weights from best epoch"""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    norm_config,
    criterion: Optional[nn.Module] = None,
    save: bool = False,
    log_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Model evaluation
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test data
        norm_config: Normalization configuration
        criterion: Loss function (defaults to MSE)
        save: Whether to save predictions
        log_path: Path to save predictions if save=True
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if criterion is None:
        criterion = nn.MSELoss()

    model.eval()
    predictions = []
    actuals = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            try:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    logger.warning("NaN loss encountered during evaluation")
                    continue
                    
                total_loss += loss.item()
                num_batches += 1

                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
            except Exception as e:
                logger.error(f"Error during evaluation batch: {e}")
                continue

    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Denormalize if needed
    if norm_config.yield_scaler is not None:
        predictions_denorm = norm_config.denormalize_yield(predictions)
        actuals_denorm = norm_config.denormalize_yield(actuals)
    else:
        predictions_denorm = predictions
        actuals_denorm = actuals

    # Calculate metrics
    metrics = {
        'r2': r2_score(actuals_denorm, predictions_denorm),
        'mse': mean_squared_error(actuals_denorm, predictions_denorm),
        'rmse': math.sqrt(mean_squared_error(actuals_denorm, predictions_denorm)),
        'mae': mean_absolute_error(actuals_denorm, predictions_denorm),
        'corr': pearsonr(actuals_denorm, predictions_denorm)[0],
        'avg_loss': total_loss / max(num_batches, 1),
        'predictions': predictions_denorm,
        'actuals': actuals_denorm
    }

    # Log metrics
    logger.info(f"Evaluation Metrics - R2: {metrics['r2']:.4f}, "
                f"MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}, "
                f"MAE: {metrics['mae']:.4f}, Pearson: {metrics['corr']:.4f}")

    # Save predictions if requested
    if save and log_path is not None:
        try:
            df_pred = pd.DataFrame({
                'Actual': metrics['actuals'],
                'Predicted': metrics['predictions']
            })
            pred_path = Path(log_path).with_suffix('.predictions.csv')
            df_pred.to_csv(pred_path, index=False)
            logger.info(f"Saved predictions to {pred_path}")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")

    return metrics

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    norm_config = None,
    log_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Training function 
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        lr: Learning rate
        norm_config: Normalization configuration
        log_path: Path to save logs and models
        
    Returns:
        Dictionary containing final evaluation metrics
    """
    # Setup training components
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True)
    
    # Training state
    best_test_loss = float('inf')
    best_epoch = 0
    training_log = []

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info("="*60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        # Training loop
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            try:
                # Skip empty batches
                if images.size(0) == 0:
                    logger.warning(f"Skipping empty batch at epoch {epoch+1}, batch {batch_idx}")
                    continue

                images, targets = images.to(device), targets.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Handle NaN loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss at epoch {epoch+1}, batch {batch_idx}")
                    continue
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1

                # Periodic memory cleanup
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error during training batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

        # Calculate average training loss
        avg_train_loss = train_loss / max(num_batches, 1) if num_batches > 0 else 0.0

        # Evaluation
        test_results = evaluate(model, test_loader, norm_config, criterion)
        test_loss = test_results['avg_loss']

        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Early stopping check
        early_stopping(test_loss, model)
        
        # Log epoch results
        log_entry = (
            f"Epoch {epoch+1}/{epochs}\n"
            f"Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}\n"
            f"MSE: {test_results['mse']:.4f} | R²: {test_results['r2']:.4f} | "
            f"MAE: {test_results['mae']:.4f} | Pearson: {test_results['corr']:.4f}\n"
            f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            + "-"*60 + "\n"
        )
        logger.info(log_entry)
        training_log.append(log_entry)

        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            
            # Save best model
            if log_path is not None:
                try:
                    model_path = Path(log_path).with_suffix('.best_model.pth')
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved best model to {model_path}")
                except Exception as e:
                    logger.error(f"Error saving model: {e}")

        # Check for early stopping
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    # Restore best weights if early stopping was used
    early_stopping.restore_model(model)

    # Final summary
    summary = (
        f"Training Complete\n"
        f"Best Model: Epoch {best_epoch} | Lowest Test Loss: {best_test_loss:.4f}\n"
        + "="*60 + "\n"
    )
    logger.info(summary)
    training_log.append(summary)

    # Save training log
    if log_path is not None:
        try:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.writelines(training_log)
            logger.info(f"Saved training log to {log_path}")
        except Exception as e:
            logger.error(f"Error saving training log: {e}")

    # Final evaluation
    final_metrics = evaluate(
        model, 
        test_loader, 
        norm_config,
        criterion, 
        save=True,
        log_path=log_path
    )

    return final_metrics

def train_model_transfer_learning(
    model_path: str,
    finetune_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    training_test_loader: torch.utils.data.DataLoader,
    finetune_epochs: int = 10,
    finetune_lr: float = 1e-5,
    log_path: Optional[str] = None,
    norm_config = None,
    config: Optional[Dict] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Enhanced transfer learning
    
    Args:
        model_path: Path to pre-trained model
        finetune_loader: DataLoader for fine-tuning data
        test_loader: DataLoader for test data
        training_test_loader: DataLoader for source domain test data
        finetune_epochs: Number of fine-tuning epochs
        finetune_lr: Learning rate for fine-tuning
        log_path: Path to save logs
        norm_config: Normalization configuration
        config: Additional configuration dictionary
        
    Returns:
        Tuple of (fine-tuning results, source domain test results)
    """
    logger.info("\n Starting Transfer Learning Process")
    
    # Validate model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")

    # Initialize model
    model = CNNRegressor(
        input_channels=config['input_channels'],
        dropout_rate=0.2
    ).to(device)

    try:
        # Load pre-trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f" Successfully loaded pre-trained weights from {model_path}")
    except Exception as e:
        logger.error(f" Error loading pre-trained weights: {e}")
        raise

    # Fine-tuning phase
    logger.info("\n Starting Fine-tuning Phase")
    finetune_results = train_model(
        model,
        finetune_loader,
        test_loader,
        epochs=finetune_epochs,
        lr=finetune_lr,
        norm_config=norm_config,
        log_path=log_path
    )

    # Evaluation on source domain
    logger.info("\n Evaluating on Source Domain")
    source_results = evaluate(
        model,
        training_test_loader,
        norm_config,
        criterion=nn.MSELoss()
    )

    return finetune_results, source_results