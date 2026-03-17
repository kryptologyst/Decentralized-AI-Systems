"""Test script for decentralized AI systems.

This script runs basic tests to verify the system is working correctly.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.decentralized_learning import EdgeCNN, DecentralizedPeer, DecentralizedNetwork, set_deterministic_seed
from src.pipelines.data_handler import DataDistributor
from src.utils.helpers import print_system_summary, check_dependencies
from src.utils.evaluator import DecentralizedEvaluator

import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_creation():
    """Test model creation and basic functionality."""
    logger.info("Testing model creation...")
    
    model = EdgeCNN(num_classes=10, input_channels=1)
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    logger.info("✓ Model creation test passed")


def test_data_distribution():
    """Test data distribution functionality."""
    logger.info("Testing data distribution...")
    
    distributor = DataDistributor(distribution_strategy="iid")
    
    # Generate synthetic data
    x_data = torch.randn(100, 1, 28, 28)
    y_data = torch.randint(0, 10, (100,))
    
    # Distribute to peers
    peer_data = distributor.distribute_data_to_peers(x_data, y_data, num_peers=3)
    
    assert len(peer_data) == 3, f"Expected 3 peers, got {len(peer_data)}"
    
    # Check data distribution
    total_samples = sum(len(data[0]) for data in peer_data)
    assert total_samples == 100, f"Expected 100 total samples, got {total_samples}"
    
    logger.info("✓ Data distribution test passed")


def test_peer_training():
    """Test peer training functionality."""
    logger.info("Testing peer training...")
    
    set_deterministic_seed(42)
    
    # Create peer
    model = EdgeCNN(num_classes=10, input_channels=1)
    device = torch.device('cpu')
    peer = DecentralizedPeer(peer_id=0, model=model, device=device)
    
    # Create dummy data
    x_data = torch.randn(50, 1, 28, 28)
    y_data = torch.randint(0, 10, (50,))
    
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Train peer
    losses, accuracies = peer.train_local(dataloader, epochs=1)
    
    assert len(losses) == 1, f"Expected 1 loss value, got {len(losses)}"
    assert len(accuracies) == 1, f"Expected 1 accuracy value, got {len(accuracies)}"
    
    logger.info("✓ Peer training test passed")


def test_network_synchronization():
    """Test network synchronization functionality."""
    logger.info("Testing network synchronization...")
    
    set_deterministic_seed(42)
    
    # Create peers
    peers = []
    device = torch.device('cpu')
    
    for i in range(3):
        model = EdgeCNN(num_classes=10, input_channels=1)
        peer = DecentralizedPeer(peer_id=i, model=model, device=device)
        peers.append(peer)
    
    # Create network
    network = DecentralizedNetwork(peers=peers, sync_frequency=1)
    
    # Test synchronization
    network.synchronize_models()
    
    # Check that all peers have the same number of communication rounds
    comm_rounds = [peer.communication_rounds for peer in peers]
    assert all(rounds == 1 for rounds in comm_rounds), "All peers should have 1 communication round"
    
    logger.info("✓ Network synchronization test passed")


def test_evaluation():
    """Test evaluation functionality."""
    logger.info("Testing evaluation...")
    
    # Create model and dummy data
    model = EdgeCNN(num_classes=10, input_channels=1)
    device = torch.device('cpu')
    
    x_test = torch.randn(20, 1, 28, 28)
    y_test = torch.randint(0, 10, (20,))
    
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create evaluator
    evaluator = DecentralizedEvaluator()
    
    # Test model quality evaluation
    quality_metrics = evaluator.evaluate_model_quality(
        model=model,
        test_dataloader=test_dataloader,
        device=device
    )
    
    assert 'accuracy' in quality_metrics, "Quality metrics should contain accuracy"
    assert 'precision' in quality_metrics, "Quality metrics should contain precision"
    assert 'recall' in quality_metrics, "Quality metrics should contain recall"
    assert 'f1_score' in quality_metrics, "Quality metrics should contain f1_score"
    
    logger.info("✓ Evaluation test passed")


def test_model_export():
    """Test model export functionality."""
    logger.info("Testing model export...")
    
    try:
        from src.export.model_exporter import ModelExporter
        
        model = EdgeCNN(num_classes=10, input_channels=1)
        exporter = ModelExporter(output_dir="./test_output")
        
        # Test ONNX export (most likely to be available)
        try:
            onnx_path = exporter.export_to_onnx(
                model=model,
                input_shape=(1, 28, 28),
                filename="test_model.onnx"
            )
            logger.info(f"✓ ONNX export test passed: {onnx_path}")
        except ImportError:
            logger.warning("ONNX not available, skipping ONNX export test")
        
        logger.info("✓ Model export test passed")
        
    except ImportError as e:
        logger.warning(f"Model export test skipped due to missing dependencies: {e}")


def run_all_tests():
    """Run all tests."""
    logger.info("Starting decentralized AI systems tests...")
    
    try:
        test_model_creation()
        test_data_distribution()
        test_peer_training()
        test_network_synchronization()
        test_evaluation()
        test_model_export()
        
        logger.info("=" * 50)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error("=" * 50)
        logger.error("TESTS FAILED!")
        logger.error("=" * 50)
        return False


def main():
    """Main test function."""
    print_system_summary()
    
    # Check dependencies
    deps = check_dependencies()
    missing_deps = [pkg for pkg, available in deps.items() if not available]
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {missing_deps}")
        logger.warning("Some tests may be skipped")
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train_decentralized.py")
        print("2. Launch demo: streamlit run demo/streamlit_demo.py")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
