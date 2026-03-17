"""Data handling and preprocessing for decentralized learning systems.

This module provides utilities for data loading, preprocessing, and distribution
across decentralized peers in edge AI systems.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pandas as pd

logger = logging.getLogger(__name__)


class DataDistributor:
    """Handles data distribution across decentralized peers.
    
    This class manages how data is split and distributed among peers,
    supporting various distribution strategies for decentralized learning.
    """
    
    def __init__(self, distribution_strategy: str = "iid") -> None:
        """Initialize the data distributor.
        
        Args:
            distribution_strategy: Strategy for data distribution ('iid', 'non_iid', 'heterogeneous').
        """
        self.distribution_strategy = distribution_strategy
        
    def load_mnist_data(
        self, 
        data_dir: str = "./data/raw",
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and preprocess MNIST dataset.
        
        Args:
            data_dir: Directory to store/load MNIST data.
            normalize: Whether to normalize pixel values to [0, 1].
            
        Returns:
            Tuple of (x_train, y_train, x_test, y_test) tensors.
        """
        logger.info("Loading MNIST dataset...")
        
        # Define transforms
        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        transform = transforms.Compose(transform_list)
        
        # Load datasets
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
        
        # Convert to tensors
        x_train = torch.stack([data[0] for data in train_dataset])
        y_train = torch.tensor([data[1] for data in train_dataset])
        x_test = torch.stack([data[0] for data in test_dataset])
        y_test = torch.tensor([data[1] for data in test_dataset])
        
        logger.info(f"MNIST loaded: {len(x_train)} train, {len(x_test)} test samples")
        
        return x_train, y_train, x_test, y_test
    
    def load_synthetic_data(
        self,
        num_samples: int = 10000,
        num_classes: int = 10,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        noise_level: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic data for testing decentralized learning.
        
        Args:
            num_samples: Total number of samples to generate.
            num_classes: Number of classes.
            input_shape: Shape of input data (channels, height, width).
            noise_level: Amount of noise to add to the data.
            
        Returns:
            Tuple of (x_train, y_train, x_test, y_test) tensors.
        """
        logger.info(f"Generating synthetic data: {num_samples} samples, {num_classes} classes")
        
        # Generate synthetic images (simplified patterns)
        x_data = []
        y_data = []
        
        samples_per_class = num_samples // num_classes
        
        for class_id in range(num_classes):
            # Create simple patterns for each class
            pattern = torch.zeros(input_shape)
            
            if input_shape[0] == 1:  # Grayscale
                # Create different patterns for each class
                if class_id % 2 == 0:
                    # Horizontal lines
                    pattern[0, class_id * 2:(class_id * 2) + 2, :] = 1.0
                else:
                    # Vertical lines
                    pattern[0, :, class_id * 2:(class_id * 2) + 2] = 1.0
            
            # Generate samples for this class
            for _ in range(samples_per_class):
                # Add noise and slight variations
                sample = pattern.clone()
                noise = torch.randn_like(sample) * noise_level
                sample = torch.clamp(sample + noise, 0, 1)
                
                x_data.append(sample)
                y_data.append(class_id)
        
        # Convert to tensors
        x_data = torch.stack(x_data)
        y_data = torch.tensor(y_data)
        
        # Split into train/test
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )
        
        logger.info(f"Synthetic data generated: {len(x_train)} train, {len(x_test)} test samples")
        
        return x_train, y_train, x_test, y_test
    
    def distribute_data_to_peers(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        num_peers: int,
        distribution_strategy: Optional[str] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Distribute data among peers according to the specified strategy.
        
        Args:
            x_data: Input data tensor.
            y_data: Target labels tensor.
            num_peers: Number of peers to distribute data to.
            distribution_strategy: Override the default distribution strategy.
            
        Returns:
            List of (x_peer, y_peer) tuples, one per peer.
        """
        strategy = distribution_strategy or self.distribution_strategy
        logger.info(f"Distributing data to {num_peers} peers using {strategy} strategy")
        
        if strategy == "iid":
            return self._distribute_iid(x_data, y_data, num_peers)
        elif strategy == "non_iid":
            return self._distribute_non_iid(x_data, y_data, num_peers)
        elif strategy == "heterogeneous":
            return self._distribute_heterogeneous(x_data, y_data, num_peers)
        else:
            raise ValueError(f"Unknown distribution strategy: {strategy}")
    
    def _distribute_iid(
        self, 
        x_data: torch.Tensor, 
        y_data: torch.Tensor, 
        num_peers: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Distribute data in an IID (Independent and Identically Distributed) manner.
        
        Each peer gets a random subset of the data with similar class distribution.
        """
        # Randomly shuffle the data
        indices = torch.randperm(len(x_data))
        x_shuffled = x_data[indices]
        y_shuffled = y_data[indices]
        
        # Split into equal parts
        chunk_size = len(x_data) // num_peers
        peer_data = []
        
        for i in range(num_peers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_peers - 1 else len(x_data)
            
            x_peer = x_shuffled[start_idx:end_idx]
            y_peer = y_shuffled[start_idx:end_idx]
            
            peer_data.append((x_peer, y_peer))
        
        return peer_data
    
    def _distribute_non_iid(
        self, 
        x_data: torch.Tensor, 
        y_data: torch.Tensor, 
        num_peers: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Distribute data in a non-IID manner.
        
        Each peer gets data from only a subset of classes, creating
        heterogeneous data distribution.
        """
        unique_classes = torch.unique(y_data).tolist()
        classes_per_peer = max(1, len(unique_classes) // num_peers)
        
        peer_data = []
        
        for peer_id in range(num_peers):
            # Assign classes to this peer
            start_class = peer_id * classes_per_peer
            end_class = min(start_class + classes_per_peer, len(unique_classes))
            
            if peer_id == num_peers - 1:  # Last peer gets remaining classes
                end_class = len(unique_classes)
            
            peer_classes = unique_classes[start_class:end_class]
            
            # Filter data for this peer's classes
            mask = torch.isin(y_data, torch.tensor(peer_classes))
            x_peer = x_data[mask]
            y_peer = y_data[mask]
            
            peer_data.append((x_peer, y_peer))
        
        return peer_data
    
    def _distribute_heterogeneous(
        self, 
        x_data: torch.Tensor, 
        y_data: torch.Tensor, 
        num_peers: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Distribute data heterogeneously with varying amounts per peer.
        
        Some peers get more data than others, simulating real-world
        heterogeneity in edge devices.
        """
        # Create different data sizes for each peer
        total_samples = len(x_data)
        peer_sizes = []
        
        # Generate random sizes (some peers have more data)
        for i in range(num_peers):
            if i < num_peers // 2:
                # First half gets more data
                size = int(total_samples * 0.7 / (num_peers // 2))
            else:
                # Second half gets less data
                size = int(total_samples * 0.3 / (num_peers - num_peers // 2))
            peer_sizes.append(size)
        
        # Adjust to match total
        peer_sizes[-1] += total_samples - sum(peer_sizes)
        
        # Shuffle data
        indices = torch.randperm(len(x_data))
        x_shuffled = x_data[indices]
        y_shuffled = y_data[indices]
        
        peer_data = []
        start_idx = 0
        
        for size in peer_sizes:
            end_idx = start_idx + size
            x_peer = x_shuffled[start_idx:end_idx]
            y_peer = y_shuffled[start_idx:end_idx]
            peer_data.append((x_peer, y_peer))
            start_idx = end_idx
        
        return peer_data
    
    def create_dataloaders(
        self,
        peer_data: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> List[DataLoader]:
        """Create DataLoaders for each peer's data.
        
        Args:
            peer_data: List of (x_peer, y_peer) tuples.
            batch_size: Batch size for DataLoaders.
            shuffle: Whether to shuffle the data.
            
        Returns:
            List of DataLoader instances, one per peer.
        """
        dataloaders = []
        
        for i, (x_peer, y_peer) in enumerate(peer_data):
            dataset = TensorDataset(x_peer, y_peer)
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=0  # Avoid multiprocessing issues
            )
            dataloaders.append(dataloader)
            
            logger.info(f"Peer {i}: {len(x_peer)} samples, {len(dataloader)} batches")
        
        return dataloaders


class DataPreprocessor:
    """Handles data preprocessing and augmentation for edge AI systems.
    
    This class provides preprocessing utilities optimized for edge devices
    with limited computational resources.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (28, 28)) -> None:
        """Initialize the data preprocessor.
        
        Args:
            input_size: Target input size (height, width).
        """
        self.input_size = input_size
        
    def normalize_tensor(
        self, 
        tensor: torch.Tensor, 
        mean: float = 0.1307, 
        std: float = 0.3081
    ) -> torch.Tensor:
        """Normalize tensor using mean and standard deviation.
        
        Args:
            tensor: Input tensor to normalize.
            mean: Mean value for normalization.
            std: Standard deviation for normalization.
            
        Returns:
            Normalized tensor.
        """
        return (tensor - mean) / std
    
    def add_noise(
        self, 
        tensor: torch.Tensor, 
        noise_level: float = 0.1
    ) -> torch.Tensor:
        """Add Gaussian noise to tensor for robustness testing.
        
        Args:
            tensor: Input tensor.
            noise_level: Standard deviation of noise.
            
        Returns:
            Tensor with added noise.
        """
        noise = torch.randn_like(tensor) * noise_level
        return torch.clamp(tensor + noise, 0, 1)
    
    def resize_tensor(
        self, 
        tensor: torch.Tensor, 
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Resize tensor to target size using interpolation.
        
        Args:
            tensor: Input tensor.
            target_size: Target size (height, width).
            
        Returns:
            Resized tensor.
        """
        return F.interpolate(
            tensor.unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    def create_augmentation_pipeline(
        self, 
        training: bool = True
    ) -> transforms.Compose:
        """Create data augmentation pipeline for training.
        
        Args:
            training: Whether to include training-specific augmentations.
            
        Returns:
            Composed transform pipeline.
        """
        transform_list = [transforms.ToTensor()]
        
        if training:
            # Light augmentations suitable for edge devices
            transform_list.extend([
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])
        
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        return transforms.Compose(transform_list)


def create_synthetic_streaming_data(
    num_samples: int = 1000,
    num_classes: int = 5,
    sequence_length: int = 10,
    feature_dim: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic streaming data for IoT sensor simulation.
    
    Args:
        num_samples: Number of samples to generate.
        num_classes: Number of classes/labels.
        sequence_length: Length of each sequence.
        feature_dim: Dimension of each feature vector.
        
        Returns:
            Tuple of (features, labels) arrays.
    """
    logger.info(f"Generating synthetic streaming data: {num_samples} samples")
    
    features = []
    labels = []
    
    for i in range(num_samples):
        # Generate random class
        label = np.random.randint(0, num_classes)
        
        # Generate sequence based on class
        if label == 0:
            # Sine wave pattern
            t = np.linspace(0, 4 * np.pi, sequence_length)
            sequence = np.sin(t).reshape(-1, 1)
        elif label == 1:
            # Random walk pattern
            sequence = np.cumsum(np.random.randn(sequence_length, 1), axis=0)
        elif label == 2:
            # Step function
            sequence = np.ones((sequence_length, 1)) * np.random.choice([-1, 1])
        else:
            # Random pattern
            sequence = np.random.randn(sequence_length, 1)
        
        # Expand to feature dimension
        feature_vector = np.tile(sequence, (1, feature_dim))
        
        # Add some noise
        feature_vector += np.random.randn(sequence_length, feature_dim) * 0.1
        
        features.append(feature_vector)
        labels.append(label)
    
    return np.array(features), np.array(labels)
