"""Main training script for decentralized AI systems.

This script demonstrates the complete decentralized learning pipeline
including data distribution, peer training, model synchronization, and evaluation.
"""

import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import our modules
from src.models.decentralized_learning import (
    EdgeCNN, DecentralizedPeer, DecentralizedNetwork, set_deterministic_seed
)
from src.pipelines.data_handler import DataDistributor
from src.export.model_exporter import ModelExporter, ModelOptimizer
from src.utils.evaluator import DecentralizedEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_peers(
    num_peers: int,
    device: torch.device,
    config: Dict[str, Any]
) -> list:
    """Create decentralized peers.
    
    Args:
        num_peers: Number of peers to create.
        device: Device to run training on.
        config: Configuration dictionary.
        
    Returns:
        List of DecentralizedPeer instances.
    """
    peers = []
    
    for peer_id in range(num_peers):
        model = EdgeCNN(
            num_classes=config['model']['num_classes'],
            input_channels=config['model']['input_channels']
        )
        
        peer = DecentralizedPeer(
            peer_id=peer_id,
            model=model,
            device=device,
            learning_rate=config['training']['learning_rate'],
            batch_size=config['training']['batch_size']
        )
        
        peers.append(peer)
    
    return peers


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Decentralized AI Systems Training')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set deterministic seed
    set_deterministic_seed(config['training']['seed'])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    data_distributor = DataDistributor(
        distribution_strategy=config['data']['distribution_strategy']
    )
    
    # Load and distribute data
    logger.info("Loading and distributing data...")
    
    if config['data']['use_synthetic']:
        x_train, y_train, x_test, y_test = data_distributor.load_synthetic_data(
            num_samples=config['data']['num_samples'],
            num_classes=config['model']['num_classes'],
            input_shape=(config['model']['input_channels'], 28, 28),
            noise_level=config['data']['noise_level']
        )
    else:
        x_train, y_train, x_test, y_test = data_distributor.load_mnist_data(
            data_dir=config['data']['data_dir']
        )
    
    # Distribute data to peers
    peer_data = data_distributor.distribute_data_to_peers(
        x_train, y_train,
        num_peers=config['network']['num_peers'],
        distribution_strategy=config['data']['distribution_strategy']
    )
    
    # Create data loaders
    peer_dataloaders = data_distributor.create_dataloaders(
        peer_data,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    # Create test data loader
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Create peers
    logger.info(f"Creating {config['network']['num_peers']} decentralized peers...")
    peers = create_peers(
        num_peers=config['network']['num_peers'],
        device=device,
        config=config
    )
    
    # Create decentralized network
    network = DecentralizedNetwork(
        peers=peers,
        sync_frequency=config['network']['sync_frequency'],
        sync_strategy=config['network']['sync_strategy']
    )
    
    # Train the network
    logger.info("Starting decentralized training...")
    training_metrics = network.train_network(
        peer_dataloaders=peer_dataloaders,
        test_dataloader=test_dataloader,
        total_rounds=config['training']['total_rounds'],
        local_epochs=config['training']['local_epochs']
    )
    
    # Get network statistics
    network_stats = network.get_network_statistics()
    
    # Export models
    logger.info("Exporting trained models...")
    exporter = ModelExporter(output_dir=str(output_dir / "models"))
    
    # Export the first peer's model as representative
    representative_model = peers[0].model
    exported_models = exporter.export_all_formats(
        model=representative_model,
        input_shape=(config['model']['input_channels'], 28, 28),
        model_name="decentralized_model"
    )
    
    # Optimize model for edge deployment
    logger.info("Optimizing model for edge deployment...")
    optimizer = ModelOptimizer()
    
    # Create optimized versions
    quantized_model = optimizer.quantize_model(
        representative_model,
        x_train[:100],  # Use first 100 samples for calibration
        method="dynamic"
    )
    
    pruned_model = optimizer.prune_model(
        representative_model,
        sparsity=config['optimization']['pruning_sparsity'],
        method="magnitude"
    )
    
    # Export optimized models
    optimized_models = exporter.export_all_formats(
        model=quantized_model,
        input_shape=(config['model']['input_channels'], 28, 28),
        model_name="quantized_model"
    )
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluator = DecentralizedEvaluator(output_dir=str(output_dir / "evaluation"))
    
    # Evaluate model quality
    quality_metrics = evaluator.evaluate_model_quality(
        model=representative_model,
        test_dataloader=test_dataloader,
        device=device,
        class_names=[str(i) for i in range(config['model']['num_classes'])]
    )
    
    # Evaluate inference efficiency
    efficiency_metrics = evaluator.evaluate_inference_efficiency(
        model=representative_model,
        input_shape=(config['model']['input_channels'], 28, 28),
        device=device,
        num_runs=config['evaluation']['num_runs'],
        batch_sizes=config['evaluation']['batch_sizes']
    )
    
    # Evaluate communication overhead
    communication_metrics = evaluator.evaluate_communication_overhead(
        network_stats=network_stats,
        model_size_mb=efficiency_metrics['model_size_mb'],
        sync_frequency=config['network']['sync_frequency']
    )
    
    # Evaluate robustness
    robustness_metrics = evaluator.evaluate_robustness(
        model=representative_model,
        test_dataloader=test_dataloader,
        device=device,
        noise_levels=config['evaluation']['noise_levels']
    )
    
    # Create evaluation report
    report_path = evaluator.create_evaluation_report(
        quality_metrics=quality_metrics,
        efficiency_metrics=efficiency_metrics,
        communication_metrics=communication_metrics,
        robustness_metrics=robustness_metrics,
        filename="decentralized_training_report.txt"
    )
    
    # Create visualization plots
    plot_paths = evaluator.plot_evaluation_results(
        quality_metrics=quality_metrics,
        efficiency_metrics=efficiency_metrics,
        communication_metrics=communication_metrics,
        robustness_metrics=robustness_metrics
    )
    
    # Save training metrics
    import json
    with open(output_dir / "training_metrics.json", 'w') as f:
        json.dump({
            'training_metrics': training_metrics,
            'network_stats': network_stats,
            'exported_models': exported_models,
            'optimized_models': optimized_models,
            'plot_paths': plot_paths
        }, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("DECENTRALIZED TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Final Network Accuracy: {network_stats['final_network_accuracy']:.2f}%")
    logger.info(f"Total Communication Rounds: {network_stats['total_communication_rounds']}")
    logger.info(f"Total Peers: {network_stats['total_peers']}")
    logger.info(f"Model Size: {efficiency_metrics['model_size_mb']:.2f} MB")
    logger.info(f"Exported Models: {list(exported_models.keys())}")
    logger.info(f"Evaluation Report: {report_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
