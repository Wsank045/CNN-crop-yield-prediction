import yaml
import os
from pathlib import Path
from experiments import (
    run_baseline_experiment,
    run_mixed_experiment,
    run_lofo_experiment,
    run_tl_experiment,
    run_cluster_experiment
)

def load_config(config_path='config.yaml'):
    """Safely load configuration file with validation"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required config sections
        required_sections = ['batch_size', 'num_workers', 'num_epochs', 
                           'finetune_sizes', 'num_clusters']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

def validate_data_paths(field_paths):
    """Verify all data files exist before running experiments"""
    missing_files = []
    for field_name, path in field_paths.items():
        if not Path(path).exists():
            missing_files.append((field_name, path))
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing data files:\n" + 
            "\n".join([f"{name}: {path}" for name, path in missing_files])
        )

def main():
    print("CROP YIELD PREDICTION PIPELINE")
    
    try:
        # Load and validate config
        config = load_config()
        
        
        field_paths = {
            'Alb_pp': 'data/alberta_pp_5m_yield.pkl',
            'Man_eds': 'data/man_eds_5m_yield.pkl',
            'Man_home': 'data/man_home_5m_yield.pkl'
        }
        
       
        validate_data_paths(field_paths)

        # Field combinations
        combos = {
            'all': field_paths,
            'home_eds': {k: v for k, v in field_paths.items() if k != 'Alb_pp'},
            'home_alb': {k: v for k, v in field_paths.items() if k != 'Man_eds'},
            'eds_alb': {k: v for k, v in field_paths.items() if k != 'Man_home'},
        }

        # Create results directory if it doesn't exist
        os.makedirs(config.get('results_dir', 'results'), exist_ok=True)

        
        # 1. Baseline Experiment
        print("\n=== RUNNING BASELINE EXPERIMENTS ===")
        run_baseline_experiment(field_paths, config=config)

        # 2. Mixed Experiment
        print("\n=== RUNNING MIXED EXPERIMENTS ===")
        for name, fields in combos.items():
            try:
                run_mixed_experiment(fields, config=config)
            except Exception as e:
                print(f"Mixed experiment failed for {name}: {e}")

        # 3. LOFO Experiment
        print("\n=== RUNNING LOFO EXPERIMENTS ===")
        for name, fields in combos.items():
            for target_field in fields:
                if len(fields) > 1:
                    try:
                        run_lofo_experiment(fields, target_field, config=config)
                    except Exception as e:
                        print(f"LOFO failed for {target_field} in {name}: {e}")

        # 4. Transfer Learning Experiment
        print("\n=== RUNNING TRANSFER LEARNING EXPERIMENTS ===")
        for name, fields in combos.items():
            for target_field in fields:
                if len(fields) > 1:
                    for size in config['finetune_sizes']:
                        try:
                            run_tl_experiment(
                                field_paths=fields,
                                target_field=target_field,
                                config=config,
                                finetune_size=size
                            )
                        except Exception as e:
                            print(f"TL failed for {target_field} at size {size}: {e}")

        
        # 5. Cluster Experiments
        print("\n=== RUNNING CLUSTER EXPERIMENTS ===")
        for field_name, path in field_paths.items():
            for cluster_id in range(config['num_clusters']):
                try:
                    run_cluster_experiment(
                        field_path=path,
                        target_field=field_name,
                        config=config,
                        cluster_id=cluster_id
                    )
                except Exception as e:
                    print(f"Cluster error for {field_name} cluster {cluster_id}: {e}")

        # 6. Reverse Cluster Experiments
        print("\n=== RUNNING REVERSE CLUSTER EXPERIMENTS ===")
        for field_name, path in field_paths.items():
            for cluster_id in range(config['num_clusters']):
                try:
                    run_cluster_experiment(
                        field_path=path,
                        target_field=field_name,
                        config=config,
                        rev_cluster=True,
                        cluster_id=cluster_id
                    )
                except Exception as e:
                    print(f"Reverse cluster error for {field_name} cluster {cluster_id}: {e}")

    except Exception as e:
        print(f"Fatal error in pipeline: {e}")
        raise

if __name__ == '__main__':
    main()