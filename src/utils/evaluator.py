"""Evaluation and metrics for decentralized learning systems.

This module provides comprehensive evaluation utilities for measuring
model quality, efficiency, and communication overhead in decentralized learning.
"""

import logging
import time
import psutil
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class DecentralizedEvaluator:
    """Comprehensive evaluator for decentralized learning systems.
    
    This class provides utilities for evaluating model quality, efficiency,
    and communication metrics in decentralized learning scenarios.
    """
    
    def __init__(self, output_dir: str = "./assets/evaluation") -> None:
        """Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_model_quality(
        self,
        model: nn.Module,
        test_dataloader: DataLoader,
        device: torch.device,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate model quality metrics.
        
        Args:
            model: PyTorch model to evaluate.
            test_dataloader: Test data loader.
            device: Device to run evaluation on.
            class_names: Optional class names for detailed reporting.
            
        Returns:
            Dictionary containing quality metrics.
        """
        logger.info("Evaluating model quality...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets, predictions, average=None
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support': support,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities,
        }
        
        if class_names:
            metrics['class_names'] = class_names
            metrics['classification_report'] = classification_report(
                targets, predictions, target_names=class_names
            )
        
        logger.info(f"Model quality evaluation completed. Accuracy: {accuracy:.4f}")
        return metrics
    
    def evaluate_inference_efficiency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: torch.device,
        num_runs: int = 100,
        batch_sizes: List[int] = [1, 8, 16, 32],
    ) -> Dict[str, Any]:
        """Evaluate inference efficiency metrics.
        
        Args:
            model: PyTorch model to evaluate.
            input_shape: Input tensor shape (excluding batch dimension).
            device: Device to run evaluation on.
            num_runs: Number of inference runs for benchmarking.
            batch_sizes: List of batch sizes to test.
            
        Returns:
            Dictionary containing efficiency metrics.
        """
        logger.info("Evaluating inference efficiency...")
        
        model.eval()
        efficiency_metrics = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape).to(device)
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Benchmark runs
            times = []
            memory_usage = []
            
            for _ in range(num_runs):
                # Measure memory before inference
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    memory_before = torch.cuda.memory_allocated(device)
                
                # Time inference
                start_time = time.time()
                with torch.no_grad():
                    output = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Measure memory after inference
                if device.type == 'cuda':
                    memory_after = torch.cuda.memory_allocated(device)
                    memory_usage.append(memory_after - memory_before)
                
                times.append(end_time - start_time)
            
            times = np.array(times)
            
            batch_metrics = {
                'mean_latency': np.mean(times),
                'std_latency': np.std(times),
                'p50_latency': np.percentile(times, 50),
                'p95_latency': np.percentile(times, 95),
                'p99_latency': np.percentile(times, 99),
                'throughput_fps': batch_size / np.mean(times),
                'throughput_samples_per_sec': batch_size / np.mean(times),
            }
            
            if memory_usage:
                batch_metrics.update({
                    'mean_memory_mb': np.mean(memory_usage) / (1024 * 1024),
                    'peak_memory_mb': np.max(memory_usage) / (1024 * 1024),
                })
            
            efficiency_metrics[f'batch_{batch_size}'] = batch_metrics
        
        # Model size metrics
        model_size_mb = self._calculate_model_size(model)
        efficiency_metrics['model_size_mb'] = model_size_mb
        
        logger.info("Inference efficiency evaluation completed")
        return efficiency_metrics
    
    def evaluate_communication_overhead(
        self,
        network_stats: Dict[str, Any],
        model_size_mb: float,
        sync_frequency: int,
    ) -> Dict[str, Any]:
        """Evaluate communication overhead in decentralized learning.
        
        Args:
            network_stats: Network statistics from DecentralizedNetwork.
            model_size_mb: Size of the model in MB.
            sync_frequency: Frequency of synchronization.
            
        Returns:
            Dictionary containing communication metrics.
        """
        logger.info("Evaluating communication overhead...")
        
        total_rounds = network_stats['total_communication_rounds']
        total_peers = network_stats['total_peers']
        total_overhead = network_stats['total_communication_overhead']
        
        # Calculate communication metrics
        total_data_transferred_mb = total_overhead * model_size_mb
        avg_data_per_round_mb = total_data_transferred_mb / total_rounds if total_rounds > 0 else 0
        avg_data_per_peer_mb = total_data_transferred_mb / total_peers if total_peers > 0 else 0
        
        # Estimate bandwidth requirements (assuming different network speeds)
        network_speeds = {
            'wifi_6': 1200,  # Mbps
            'wifi_5': 433,   # Mbps
            '4g': 100,       # Mbps
            '3g': 3,         # Mbps
        }
        
        bandwidth_requirements = {}
        for network_type, speed_mbps in network_speeds.items():
            speed_mb_per_sec = speed_mbps / 8  # Convert to MB/s
            time_per_sync_sec = avg_data_per_round_mb / speed_mb_per_sec
            bandwidth_requirements[network_type] = {
                'sync_time_seconds': time_per_sync_sec,
                'sync_time_minutes': time_per_sync_sec / 60,
                'feasible': time_per_sync_sec < 60,  # Feasible if sync takes less than 1 minute
            }
        
        communication_metrics = {
            'total_data_transferred_mb': total_data_transferred_mb,
            'avg_data_per_round_mb': avg_data_per_round_mb,
            'avg_data_per_peer_mb': avg_data_per_peer_mb,
            'total_sync_operations': total_overhead,
            'sync_frequency': sync_frequency,
            'bandwidth_requirements': bandwidth_requirements,
            'communication_efficiency': {
                'data_per_accuracy_point': total_data_transferred_mb / max(network_stats.get('final_network_accuracy', 1), 1),
                'sync_overhead_ratio': total_overhead / (total_rounds * total_peers) if total_rounds > 0 and total_peers > 0 else 0,
            }
        }
        
        logger.info("Communication overhead evaluation completed")
        return communication_metrics
    
    def evaluate_robustness(
        self,
        model: nn.Module,
        test_dataloader: DataLoader,
        device: torch.device,
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
    ) -> Dict[str, Any]:
        """Evaluate model robustness to noise and perturbations.
        
        Args:
            model: PyTorch model to evaluate.
            test_dataloader: Test data loader.
            device: Device to run evaluation on.
            noise_levels: List of noise levels to test.
            
        Returns:
            Dictionary containing robustness metrics.
        """
        logger.info("Evaluating model robustness...")
        
        robustness_metrics = {}
        
        for noise_level in noise_levels:
            logger.info(f"Testing robustness with noise level: {noise_level}")
            
            accuracies = []
            
            model.eval()
            with torch.no_grad():
                for data, target in test_dataloader:
                    data, target = data.to(device), target.to(device)
                    
                    # Add noise to input
                    if noise_level > 0:
                        noise = torch.randn_like(data) * noise_level
                        data = torch.clamp(data + noise, 0, 1)
                    
                    output = model(data)
                    predictions = output.argmax(dim=1)
                    accuracy = (predictions == target).float().mean().item()
                    accuracies.append(accuracy)
            
            avg_accuracy = np.mean(accuracies)
            robustness_metrics[f'noise_{noise_level}'] = {
                'accuracy': avg_accuracy,
                'accuracy_drop': robustness_metrics.get('noise_0.0', {}).get('accuracy', 1.0) - avg_accuracy,
            }
        
        # Calculate robustness score
        baseline_accuracy = robustness_metrics.get('noise_0.0', {}).get('accuracy', 1.0)
        robustness_score = 0
        for noise_level in noise_levels[1:]:  # Skip baseline
            key = f'noise_{noise_level}'
            if key in robustness_metrics:
                accuracy_drop = baseline_accuracy - robustness_metrics[key]['accuracy']
                robustness_score += accuracy_drop / noise_level
        
        robustness_metrics['robustness_score'] = robustness_score
        robustness_metrics['baseline_accuracy'] = baseline_accuracy
        
        logger.info("Robustness evaluation completed")
        return robustness_metrics
    
    def create_evaluation_report(
        self,
        quality_metrics: Dict[str, Any],
        efficiency_metrics: Dict[str, Any],
        communication_metrics: Dict[str, Any],
        robustness_metrics: Dict[str, Any],
        filename: str = "evaluation_report.txt",
    ) -> str:
        """Create a comprehensive evaluation report.
        
        Args:
            quality_metrics: Model quality metrics.
            efficiency_metrics: Inference efficiency metrics.
            communication_metrics: Communication overhead metrics.
            robustness_metrics: Model robustness metrics.
            filename: Output filename for the report.
            
        Returns:
            Path to the generated report.
        """
        logger.info("Creating evaluation report...")
        
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write("DECENTRALIZED AI SYSTEMS - EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Model Quality Section
            f.write("MODEL QUALITY METRICS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Accuracy: {quality_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {quality_metrics['precision']:.4f}\n")
            f.write(f"Recall: {quality_metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {quality_metrics['f1_score']:.4f}\n\n")
            
            # Efficiency Section
            f.write("INFERENCE EFFICIENCY METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model Size: {efficiency_metrics['model_size_mb']:.2f} MB\n")
            
            for batch_key, batch_metrics in efficiency_metrics.items():
                if batch_key.startswith('batch_'):
                    batch_size = batch_key.split('_')[1]
                    f.write(f"\nBatch Size {batch_size}:\n")
                    f.write(f"  Mean Latency: {batch_metrics['mean_latency']:.4f} seconds\n")
                    f.write(f"  P95 Latency: {batch_metrics['p95_latency']:.4f} seconds\n")
                    f.write(f"  Throughput: {batch_metrics['throughput_fps']:.2f} FPS\n")
                    if 'mean_memory_mb' in batch_metrics:
                        f.write(f"  Memory Usage: {batch_metrics['mean_memory_mb']:.2f} MB\n")
            
            # Communication Section
            f.write("\nCOMMUNICATION OVERHEAD METRICS\n")
            f.write("-" * 32 + "\n")
            f.write(f"Total Data Transferred: {communication_metrics['total_data_transferred_mb']:.2f} MB\n")
            f.write(f"Average Data per Round: {communication_metrics['avg_data_per_round_mb']:.2f} MB\n")
            f.write(f"Average Data per Peer: {communication_metrics['avg_data_per_peer_mb']:.2f} MB\n")
            f.write(f"Sync Operations: {communication_metrics['total_sync_operations']}\n")
            
            f.write("\nBandwidth Requirements:\n")
            for network_type, requirements in communication_metrics['bandwidth_requirements'].items():
                f.write(f"  {network_type.upper()}: {requirements['sync_time_seconds']:.2f}s "
                       f"({'Feasible' if requirements['feasible'] else 'Not Feasible'})\n")
            
            # Robustness Section
            f.write("\nROBUSTNESS METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Baseline Accuracy: {robustness_metrics['baseline_accuracy']:.4f}\n")
            f.write(f"Robustness Score: {robustness_metrics['robustness_score']:.4f}\n")
            
            for noise_key, noise_metrics in robustness_metrics.items():
                if noise_key.startswith('noise_') and noise_key != 'noise_0.0':
                    noise_level = noise_key.split('_')[1]
                    f.write(f"Noise Level {noise_level}: {noise_metrics['accuracy']:.4f} "
                           f"(Drop: {noise_metrics['accuracy_drop']:.4f})\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("END OF EVALUATION REPORT\n")
        
        logger.info(f"Evaluation report saved: {report_path}")
        return str(report_path)
    
    def plot_evaluation_results(
        self,
        quality_metrics: Dict[str, Any],
        efficiency_metrics: Dict[str, Any],
        communication_metrics: Dict[str, Any],
        robustness_metrics: Dict[str, Any],
    ) -> Dict[str, str]:
        """Create visualization plots for evaluation results.
        
        Args:
            quality_metrics: Model quality metrics.
            efficiency_metrics: Inference efficiency metrics.
            communication_metrics: Communication overhead metrics.
            robustness_metrics: Model robustness metrics.
            
        Returns:
            Dictionary mapping plot names to file paths.
        """
        logger.info("Creating evaluation plots...")
        
        plot_paths = {}
        
        # Confusion Matrix
        if 'confusion_matrix' in quality_metrics:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                quality_metrics['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=quality_metrics.get('class_names', range(10)),
                yticklabels=quality_metrics.get('class_names', range(10))
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            cm_path = self.output_dir / 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['confusion_matrix'] = str(cm_path)
        
        # Latency vs Batch Size
        batch_sizes = []
        latencies = []
        throughputs = []
        
        for batch_key, batch_metrics in efficiency_metrics.items():
            if batch_key.startswith('batch_'):
                batch_size = int(batch_key.split('_')[1])
                batch_sizes.append(batch_size)
                latencies.append(batch_metrics['mean_latency'])
                throughputs.append(batch_metrics['throughput_fps'])
        
        if batch_sizes:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Latency plot
            ax1.plot(batch_sizes, latencies, 'bo-')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Mean Latency (seconds)')
            ax1.set_title('Inference Latency vs Batch Size')
            ax1.grid(True)
            
            # Throughput plot
            ax2.plot(batch_sizes, throughputs, 'ro-')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Throughput (FPS)')
            ax2.set_title('Throughput vs Batch Size')
            ax2.grid(True)
            
            plt.tight_layout()
            efficiency_path = self.output_dir / 'efficiency_metrics.png'
            plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['efficiency_metrics'] = str(efficiency_path)
        
        # Robustness plot
        noise_levels = []
        accuracies = []
        
        for noise_key, noise_metrics in robustness_metrics.items():
            if noise_key.startswith('noise_'):
                noise_level = float(noise_key.split('_')[1])
                noise_levels.append(noise_level)
                accuracies.append(noise_metrics['accuracy'])
        
        if noise_levels:
            plt.figure(figsize=(10, 6))
            plt.plot(noise_levels, accuracies, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Noise Level')
            plt.ylabel('Accuracy')
            plt.title('Model Robustness to Noise')
            plt.grid(True)
            plt.ylim(0, 1)
            
            robustness_path = self.output_dir / 'robustness_metrics.png'
            plt.savefig(robustness_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['robustness_metrics'] = str(robustness_path)
        
        logger.info(f"Created {len(plot_paths)} evaluation plots")
        return plot_paths
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Model size in MB.
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # Convert to MB
