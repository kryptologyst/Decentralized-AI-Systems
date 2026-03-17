"""Decentralized AI Systems - Core Models and Training Logic.

This module implements peer-to-peer decentralized learning without a central server,
focusing on edge device coordination and model synchronization.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import copy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed value for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EdgeCNN(nn.Module):
    """Lightweight CNN model optimized for edge devices.
    
    This model is designed for resource-constrained environments with
    minimal memory and compute requirements.
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1) -> None:
        """Initialize the EdgeCNN model.
        
        Args:
            num_classes: Number of output classes.
            input_channels: Number of input channels (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size based on input dimensions
        # For 28x28 input: 28 -> 14 -> 7 -> 3 (after 3 pooling layers)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class DecentralizedPeer:
    """Represents a single peer in the decentralized learning network.
    
    Each peer maintains its own local model and participates in peer-to-peer
    model synchronization without requiring a central server.
    """
    
    def __init__(
        self,
        peer_id: int,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ) -> None:
        """Initialize a decentralized peer.
        
        Args:
            peer_id: Unique identifier for this peer.
            model: PyTorch model to train.
            device: Device to run training on (CPU/CUDA).
            learning_rate: Learning rate for optimizer.
            batch_size: Batch size for training.
        """
        self.peer_id = peer_id
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        
        # Track training metrics
        self.training_losses: List[float] = []
        self.training_accuracies: List[float] = []
        self.communication_rounds: int = 0
        
    def train_local(
        self, 
        dataloader: DataLoader, 
        epochs: int = 1
    ) -> Tuple[List[float], List[float]]:
        """Train the local model on peer's data.
        
        Args:
            dataloader: DataLoader containing the peer's local data.
            epochs: Number of training epochs.
            
        Returns:
            Tuple of (losses, accuracies) for each epoch.
        """
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total_samples = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = 100.0 * correct / total_samples
            
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
            
            logger.info(
                f"Peer {self.peer_id} - Epoch {epoch+1}: "
                f"Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%"
            )
        
        self.training_losses.extend(epoch_losses)
        self.training_accuracies.extend(epoch_accuracies)
        
        return epoch_losses, epoch_accuracies
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model on test data.
        
        Args:
            dataloader: DataLoader containing test data.
            
        Returns:
            Tuple of (average_loss, accuracy_percentage).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total_samples
        
        return avg_loss, accuracy
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get the current model state for synchronization.
        
        Returns:
            Dictionary containing model state dict and metadata.
        """
        return {
            'state_dict': copy.deepcopy(self.model.state_dict()),
            'peer_id': self.peer_id,
            'communication_round': self.communication_rounds,
            'training_samples': len(self.training_losses),
        }
    
    def set_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Set the model state from synchronized weights.
        
        Args:
            state_dict: Model state dictionary from another peer.
        """
        self.model.load_state_dict(state_dict)
        self.communication_rounds += 1


class DecentralizedNetwork:
    """Manages a network of decentralized peers for collaborative learning.
    
    This class orchestrates peer-to-peer communication and model synchronization
    without requiring a central server.
    """
    
    def __init__(
        self,
        peers: List[DecentralizedPeer],
        sync_frequency: int = 1,
        sync_strategy: str = "average",
    ) -> None:
        """Initialize the decentralized network.
        
        Args:
            peers: List of DecentralizedPeer instances.
            sync_frequency: How often to synchronize models (in training rounds).
            sync_strategy: Strategy for model synchronization ('average', 'weighted').
        """
        self.peers = peers
        self.sync_frequency = sync_frequency
        self.sync_strategy = sync_strategy
        self.global_round = 0
        
        # Track network-wide metrics
        self.network_losses: List[float] = []
        self.network_accuracies: List[float] = []
        self.communication_overhead: List[int] = []
        
    def synchronize_models(self) -> None:
        """Synchronize models across all peers using the specified strategy.
        
        This implements peer-to-peer model averaging without a central server.
        """
        logger.info(f"Starting model synchronization (Round {self.global_round})")
        
        # Collect model states from all peers
        peer_states = [peer.get_model_state() for peer in self.peers]
        
        if self.sync_strategy == "average":
            self._average_models(peer_states)
        elif self.sync_strategy == "weighted":
            self._weighted_average_models(peer_states)
        else:
            raise ValueError(f"Unknown sync strategy: {self.sync_strategy}")
        
        # Track communication overhead (simplified: number of model exchanges)
        self.communication_overhead.append(len(self.peers) * (len(self.peers) - 1))
        
        logger.info("Model synchronization completed")
    
    def _average_models(self, peer_states: List[Dict[str, Any]]) -> None:
        """Average model weights across all peers.
        
        Args:
            peer_states: List of model states from all peers.
        """
        # Get the first peer's state dict as template
        template_state = peer_states[0]['state_dict']
        averaged_state = {}
        
        # Average each parameter across all peers
        for param_name in template_state.keys():
            param_tensors = [state['state_dict'][param_name] for state in peer_states]
            averaged_state[param_name] = torch.stack(param_tensors).mean(dim=0)
        
        # Distribute averaged weights back to all peers
        for peer in self.peers:
            peer.set_model_state(averaged_state)
    
    def _weighted_average_models(self, peer_states: List[Dict[str, Any]]) -> None:
        """Perform weighted average based on peer training samples.
        
        Args:
            peer_states: List of model states from all peers.
        """
        # Calculate weights based on training samples
        total_samples = sum(state['training_samples'] for state in peer_states)
        weights = [state['training_samples'] / total_samples for state in peer_states]
        
        # Get the first peer's state dict as template
        template_state = peer_states[0]['state_dict']
        weighted_state = {}
        
        # Weighted average each parameter
        for param_name in template_state.keys():
            param_tensors = [state['state_dict'][param_name] for state in peer_states]
            weighted_sum = sum(w * tensor for w, tensor in zip(weights, param_tensors))
            weighted_state[param_name] = weighted_sum
        
        # Distribute weighted averaged weights back to all peers
        for peer in self.peers:
            peer.set_model_state(weighted_state)
    
    def train_network(
        self,
        peer_dataloaders: List[DataLoader],
        test_dataloader: DataLoader,
        total_rounds: int = 10,
        local_epochs: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the decentralized network for multiple rounds.
        
        Args:
            peer_dataloaders: List of DataLoaders, one per peer.
            test_dataloader: DataLoader for evaluation.
            total_rounds: Total number of communication rounds.
            local_epochs: Number of local training epochs per round.
            
        Returns:
            Dictionary containing training metrics.
        """
        logger.info(f"Starting decentralized training for {total_rounds} rounds")
        
        for round_num in range(total_rounds):
            self.global_round = round_num
            
            # Local training phase
            logger.info(f"Round {round_num + 1}: Local training phase")
            for peer, dataloader in zip(self.peers, peer_dataloaders):
                peer.train_local(dataloader, epochs=local_epochs)
            
            # Synchronization phase
            if (round_num + 1) % self.sync_frequency == 0:
                self.synchronize_models()
            
            # Evaluation phase
            logger.info(f"Round {round_num + 1}: Evaluation phase")
            round_losses = []
            round_accuracies = []
            
            for peer in self.peers:
                loss, accuracy = peer.evaluate(test_dataloader)
                round_losses.append(loss)
                round_accuracies.append(accuracy)
            
            avg_loss = np.mean(round_losses)
            avg_accuracy = np.mean(round_accuracies)
            
            self.network_losses.append(avg_loss)
            self.network_accuracies.append(avg_accuracy)
            
            logger.info(
                f"Round {round_num + 1} Results: "
                f"Avg Loss={avg_loss:.4f}, Avg Accuracy={avg_accuracy:.2f}%"
            )
        
        return {
            'network_losses': self.network_losses,
            'network_accuracies': self.network_accuracies,
            'communication_overhead': self.communication_overhead,
        }
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics.
        
        Returns:
            Dictionary containing network-wide metrics and peer statistics.
        """
        peer_stats = []
        for peer in self.peers:
            peer_stats.append({
                'peer_id': peer.peer_id,
                'communication_rounds': peer.communication_rounds,
                'total_training_samples': len(peer.training_losses),
                'final_training_loss': peer.training_losses[-1] if peer.training_losses else 0.0,
                'final_training_accuracy': peer.training_accuracies[-1] if peer.training_accuracies else 0.0,
            })
        
        return {
            'total_peers': len(self.peers),
            'total_communication_rounds': self.global_round + 1,
            'sync_strategy': self.sync_strategy,
            'sync_frequency': self.sync_frequency,
            'total_communication_overhead': sum(self.communication_overhead),
            'final_network_accuracy': self.network_accuracies[-1] if self.network_accuracies else 0.0,
            'peer_statistics': peer_stats,
        }
