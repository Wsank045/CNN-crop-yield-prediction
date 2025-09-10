import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import geopandas as gpd
from shapely import wkt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

class NormalizationConfig:
    """
    Normalization configuration 
    """
    def __init__(self,
                 image_normalize_bands: bool = True,
                 yield_normalization: str = 'zscore',  # 'minmax', 'zscore', or 'none'
                 yield_scaler: Optional[object] = None,
                 yield_stats: Optional[Dict] = None):
        """
        Args:
            image_normalize_bands: Whether to normalize image bands to [0,1]
            yield_normalization: Type of yield normalization
            yield_scaler: Pre-fitted scaler for yield values
            yield_stats: Dictionary with yield statistics for denormalization
        """
        if yield_normalization not in ['minmax', 'zscore', 'none']:
            raise ValueError("yield_normalization must be 'minmax', 'zscore' or 'none'")
            
        self.image_normalize_bands = image_normalize_bands
        self.yield_normalization = yield_normalization
        self.yield_scaler = yield_scaler
        self.yield_stats = yield_stats or {}

    def fit_yield_scaler(self, yield_values: np.ndarray) -> None:
        """Fit the yield scaler on training data with validation"""
        if not isinstance(yield_values, np.ndarray):
            yield_values = np.array(yield_values)
        
        if len(yield_values.shape) == 1:
            yield_values = yield_values.reshape(-1, 1)

        if self.yield_normalization == 'minmax':
            self.yield_scaler = MinMaxScaler()
            self.yield_scaler.fit(yield_values)
            self.yield_stats = {
                'min': float(yield_values.min()),
                'max': float(yield_values.max())
            }
        elif self.yield_normalization == 'zscore':
            self.yield_scaler = StandardScaler()
            self.yield_scaler.fit(yield_values)
            self.yield_stats = {
                'mean': float(yield_values.mean()),
                'std': float(yield_values.std())
            }

    def normalize_yield(self, yield_value: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray]:
        """Normalize yield values with type checking"""
        if self.yield_scaler is None:
            return yield_value
            
        if isinstance(yield_value, torch.Tensor):
            yield_value = yield_value.cpu().numpy()
            
        if isinstance(yield_value, (int, float)):
            yield_value = np.array([[yield_value]])
        else:
            yield_value = np.array(yield_value).reshape(-1, 1)
            
        return self.yield_scaler.transform(yield_value).flatten()[0]

    def denormalize_yield(self, normalized_values: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Denormalize yield values with type checking"""
        if self.yield_scaler is None:
            return normalized_values
            
        if isinstance(normalized_values, torch.Tensor):
            normalized_values = normalized_values.cpu().numpy()
            
        normalized_values = np.array(normalized_values).reshape(-1, 1)
        return self.yield_scaler.inverse_transform(normalized_values).flatten()

class CreateDataset(Dataset):
    """ Dataset Class"""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 selected_bands: Optional[List[int]],
                 norm_config: NormalizationConfig,
                 target_size: Tuple[int, int] = (140, 140)):
        """
        Args:
            df: Input DataFrame with image data
            selected_bands: Bands to select from image data
            norm_config: Normalization configuration
            target_size: Target size for image resizing
        """
        self.df = df.reset_index(drop=True)
        self.selected_bands = selected_bands
        self.norm_config = norm_config
        self.target_size = target_size
        self.valid_indices = self._validate_samples()
        
    def _validate_samples(self) -> List[int]:
        """Identify valid samples in the dataset"""
        valid_indices = []
        for idx in range(len(self.df)):
            try:
                row = self.df.iloc[idx]
                img = row['image_data']
                if not isinstance(img, np.ndarray):
                    continue
                if img.ndim != 3:
                    continue
                valid_indices.append(idx)
            except Exception:
                continue
        return valid_indices
        
    def __len__(self) -> int:
        return len(self.valid_indices)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        try:
            real_idx = self.valid_indices[idx]
            row = self.df.iloc[real_idx]
            
            img = row['image_data'].astype(np.float32)
            yield_value = float(row['mean_yield'])
            
            # Band selection with validation
            if row['field_name'] == 'Alb_pp':
                img = img[[0, 1, 2, 3, 4], :, :]
            elif self.selected_bands is not None:
                img = img[self.selected_bands, :, :]
                
            # Validate image data
            valid_pixels = np.all(img > 0, axis=0)
            if not np.any(valid_pixels):
                raise ValueError(f"No valid pixels in sample {real_idx}")
                
            img = img * valid_pixels[np.newaxis, :, :]
            
            # Normalize bands
            if self.norm_config.image_normalize_bands:
                for band in range(img.shape[0]):
                    band_data = img[band, :, :]
                    band_min, band_max = band_data.min(), band_data.max()
                    if band_max > band_min:
                        img[band, :, :] = (band_data - band_min) / (band_max - band_min)
                    else:
                        img[band, :, :] = 0
                        
            # Normalize yield
            if self.norm_config.yield_scaler is not None:
                yield_value = self.norm_config.normalize_yield(yield_value)
                
            # Convert to tensor and resize
            image = torch.tensor(img, dtype=torch.float32)
            if image.shape[1:] != torch.Size(self.target_size):
                image = F.interpolate(image.unsqueeze(0), 
                                    size=self.target_size, 
                                    mode='bilinear', 
                                    align_corners=False).squeeze(0)
                                    
            return image, torch.tensor(yield_value, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return zero tensors of correct shape
            num_bands = len(self.selected_bands) if self.selected_bands else 1
            dummy_img = torch.zeros((num_bands, *self.target_size))
            dummy_yield = torch.tensor(0.0, dtype=torch.float32)
            return dummy_img, dummy_yield

def prepare_datasets_with_normalization(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict
) -> Tuple[CreateDataset, CreateDataset, NormalizationConfig]:
    """
    Dataset preparation with validation
    """
    # Validate input dataframes
    for df, name in [(train_df, 'train'), (test_df, 'test')]:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{name}_df must be a pandas DataFrame")
        if len(df) == 0:
            raise ValueError(f"{name}_df is empty")
            
    # Create and configure normalization
    norm_config = NormalizationConfig(
        image_normalize_bands=config.get('image_normalize_bands', True),
        yield_normalization=config.get('yield_normalization', 'zscore')
    )
    
    # Fit scaler only on training data
    if norm_config.yield_normalization != 'none':
        norm_config.fit_yield_scaler(train_df['mean_yield'].values)
        
    # Validate selected bands
    selected_bands = config.get('selected_bands')
    if selected_bands is not None:
        if not isinstance(selected_bands, list):
            raise ValueError("selected_bands must be a list or None")
            
    return (
        CreateDataset(train_df, selected_bands, norm_config),
        CreateDataset(test_df, selected_bands, norm_config),
        norm_config
    )

def load_and_split(
    field_paths: Dict[str, str],
    target_field: Optional[str] = None,
    strategy: str = 'mixed',
    test_size: float = 0.2,
    fine_tune_size: float = 0.3,
    random_state: int = 42
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    """Data loading and splitting with validation"""
    # Validate inputs
    if not isinstance(field_paths, dict) or len(field_paths) == 0:
        raise ValueError("field_paths must be a non-empty dictionary")
        
    if strategy not in ['mixed', 'lofo', 'transfer_learning']:
        raise ValueError("strategy must be 'mixed', 'lofo' or 'transfer_learning'")
        
    if strategy in ['lofo', 'transfer_learning'] and target_field is None:
        raise ValueError("target_field must be specified for lofo or transfer_learning strategies")
        
    if target_field is not None and target_field not in field_paths:
        raise KeyError(f"target_field '{target_field}' not in field_paths")

    # Load data with validation
    dfs = {}
    for field_name, file_path in field_paths.items():
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            df = pd.read_pickle(file_path)
            required_cols = ['image_data', 'mean_yield', 'polygon']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {field_name}: {missing_cols}")
                
            df['field_name'] = field_name
            dfs[field_name] = df
            
        except Exception as e:
            raise RuntimeError(f"Error loading {field_name}: {e}")

    # Implement strategies
    if strategy == 'mixed':
        min_size = min(len(df) for df in dfs.values())
        balanced_dfs = [df.sample(n=min_size, random_state=random_state) for df in dfs.values()]
        combined_df = pd.concat(balanced_dfs, ignore_index=True)
        return train_test_split(combined_df, test_size=test_size, random_state=random_state)
        
    elif strategy == 'lofo':
        test_df = dfs[target_field].copy()
        train_df = pd.concat([df for name, df in dfs.items() if name != target_field], ignore_index=True)
        return train_df, test_df
        
    elif strategy == 'transfer_learning':
        target_df = dfs[target_field]
        train_df = pd.concat([df for name, df in dfs.items() if name != target_field], ignore_index=True)
        
        # Split pretrain data
        _, pretest_df = train_test_split(train_df, test_size=0.2, random_state=random_state)
        
        # Split target data
        fine_tune_df, test_df = train_test_split(
            target_df, 
            test_size=1-fine_tune_size, 
            random_state=random_state
        )
        
        return fine_tune_df, test_df, pretest_df

def load_cluster_data(
    field_path: str,
    target_field : str,
    num_clusters: int = 5,
    rev_cluster: bool = False,
    cluster_id: Optional[int] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """cluster data loading with validation"""
    try:
        df = pd.read_pickle(field_path)
        df ['field_name'] = target_field
        
        # Process geometry
        if 'polygon' not in df.columns:
            raise ValueError("DataFrame must contain 'polygon' column")
            
        df['geometry'] = df['polygon'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf['centroid'] = gdf.geometry.centroid
        gdf['x'] = gdf.centroid.x
        gdf['y'] = gdf.centroid.y
        
        # Cluster validation
        if cluster_id is None:
            raise ValueError("cluster_id must be specified")
        if not 0 <= cluster_id < num_clusters:
            raise ValueError(f"cluster_id must be between 0 and {num_clusters-1}")
            
        # Perform clustering
        coords = gdf[['x', 'y']].values
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init='auto')
        gdf['cluster'] = kmeans.fit_predict(coords)
        
        # Split based on cluster
        if rev_cluster:
            train_df = gdf[gdf['cluster'] != cluster_id]
            test_df = gdf[gdf['cluster'] == cluster_id]
        else:
            train_df = gdf[gdf['cluster'] == cluster_id]
            test_df = gdf[gdf['cluster'] != cluster_id]
            
        return train_df, test_df
        
    except Exception as e:
        raise RuntimeError(f"Error in load_cluster_data: {e}")