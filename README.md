# Decentralized AI Systems

A comprehensive research and educational framework for implementing peer-to-peer decentralized learning systems optimized for edge AI and IoT applications.

## ⚠️ Important Disclaimer

**This is a research and educational project only.** This system is NOT intended for safety-critical applications or production use. The models and algorithms shown here are for research and educational purposes only.

## Overview

This project implements a decentralized AI system where multiple edge devices (peers) collaborate to train machine learning models without requiring a central server. Unlike traditional federated learning, this system uses peer-to-peer communication and model synchronization, making it suitable for scenarios where:

- No central server is available or desired
- Privacy and data sovereignty are critical
- Network connectivity is intermittent
- Edge devices need to operate independently

## Key Features

- **Peer-to-Peer Learning**: Decentralized model training without central coordination
- **Edge-Optimized Models**: Lightweight CNN architectures suitable for resource-constrained devices
- **Multiple Data Distribution Strategies**: IID, non-IID, and heterogeneous data distribution
- **Model Export Pipeline**: Support for ONNX, TensorFlow Lite, CoreML, and OpenVINO
- **Comprehensive Evaluation**: Quality, efficiency, communication, and robustness metrics
- **Interactive Demo**: Streamlit-based visualization and simulation
- **Edge Deployment Support**: Configurations for various edge devices

## Project Structure

```
├── src/
│   ├── models/
│   │   └── decentralized_learning.py    # Core decentralized learning implementation
│   ├── pipelines/
│   │   └── data_handler.py              # Data loading and distribution utilities
│   ├── export/
│   │   └── model_exporter.py            # Model export and optimization tools
│   ├── runtimes/
│   ├── comms/
│   └── utils/
│       └── evaluator.py                 # Comprehensive evaluation metrics
├── configs/
│   ├── training_config.yaml             # Main training configuration
│   └── device/
│       └── edge_devices.yaml            # Edge device configurations
├── scripts/
│   └── train_decentralized.py           # Main training script
├── demo/
│   └── streamlit_demo.py                # Interactive demo application
├── data/
│   ├── raw/                             # Raw data storage
│   └── processed/                       # Processed data storage
├── tests/                               # Unit tests
├── assets/                              # Generated models and results
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Project configuration
└── README.md                            # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Decentralized-AI-Systems.git
cd Decentralized-AI-Systems

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### 2. Run Training

```bash
# Basic training with default configuration
python scripts/train_decentralized.py

# Custom configuration
python scripts/train_decentralized.py --config configs/training_config.yaml --output_dir ./outputs
```

### 3. Launch Interactive Demo

```bash
# Start Streamlit demo
streamlit run demo/streamlit_demo.py
```

## Configuration

### Training Configuration

Edit `configs/training_config.yaml` to customize:

- **Model**: Architecture, number of classes, input channels
- **Data**: Dataset selection, distribution strategy, sample count
- **Training**: Learning rate, batch size, number of rounds
- **Network**: Number of peers, synchronization frequency and strategy
- **Evaluation**: Metrics, batch sizes, noise levels

### Device Configuration

Edit `configs/device/edge_devices.yaml` for target device specifications:

- Raspberry Pi 4
- NVIDIA Jetson Nano
- Android devices
- iOS devices
- Generic edge devices

## Usage Examples

### Basic Decentralized Training

```python
from src.models.decentralized_learning import EdgeCNN, DecentralizedPeer, DecentralizedNetwork
from src.pipelines.data_handler import DataDistributor

# Set up data distribution
distributor = DataDistributor(distribution_strategy="non_iid")
x_train, y_train, x_test, y_test = distributor.load_synthetic_data()

# Create peers
peers = []
for i in range(3):
    model = EdgeCNN(num_classes=10)
    peer = DecentralizedPeer(peer_id=i, model=model, device=device)
    peers.append(peer)

# Create network
network = DecentralizedNetwork(peers=peers, sync_frequency=1)

# Train network
metrics = network.train_network(peer_dataloaders, test_dataloader, total_rounds=10)
```

### Model Export

```python
from src.export.model_exporter import ModelExporter

# Export to multiple formats
exporter = ModelExporter()
exported_models = exporter.export_all_formats(
    model=trained_model,
    input_shape=(1, 28, 28),
    model_name="decentralized_model"
)
```

### Evaluation

```python
from src.utils.evaluator import DecentralizedEvaluator

# Comprehensive evaluation
evaluator = DecentralizedEvaluator()
quality_metrics = evaluator.evaluate_model_quality(model, test_dataloader, device)
efficiency_metrics = evaluator.evaluate_inference_efficiency(model, input_shape, device)
```

## Key Components

### DecentralizedPeer

Represents a single peer in the decentralized network:
- Local model training
- Model state synchronization
- Performance tracking

### DecentralizedNetwork

Manages the peer-to-peer network:
- Model synchronization strategies (average, weighted)
- Communication overhead tracking
- Network-wide metrics

### DataDistributor

Handles data distribution across peers:
- IID (Independent and Identically Distributed)
- Non-IID (Non-Independent and Identically Distributed)
- Heterogeneous distribution

### ModelExporter

Exports models to edge-optimized formats:
- ONNX for cross-platform deployment
- TensorFlow Lite for mobile devices
- CoreML for iOS devices
- OpenVINO for Intel hardware

## Evaluation Metrics

### Model Quality
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix and per-class metrics
- Classification reports

### Efficiency Metrics
- Inference latency (mean, P95, P99)
- Throughput (FPS, samples/second)
- Memory usage
- Model size

### Communication Overhead
- Total data transferred
- Average data per round/peer
- Bandwidth requirements
- Sync operation efficiency

### Robustness
- Noise tolerance
- Accuracy degradation analysis
- Robustness scoring

## Edge Deployment

### Supported Platforms

- **Raspberry Pi 4**: ONNX Runtime, TensorFlow Lite
- **NVIDIA Jetson**: TensorRT, ONNX Runtime
- **Android**: TensorFlow Lite, ONNX Runtime
- **iOS**: CoreML, ONNX Runtime
- **Generic Edge**: ONNX Runtime, OpenVINO

### Optimization Techniques

- **Quantization**: Dynamic and static quantization
- **Pruning**: Magnitude and random pruning
- **Model Fusion**: Operation fusion for efficiency
- **Hardware-Specific**: TensorRT, CoreML optimizations

## Interactive Demo

The Streamlit demo provides:

1. **Training Simulation**: Real-time visualization of decentralized training
2. **Model Performance**: Comprehensive evaluation metrics and visualizations
3. **Communication Analysis**: Network overhead and efficiency analysis
4. **Edge Deployment**: Device-specific deployment simulation

Access the demo at: `http://localhost:8501`

## Research Applications

This framework is suitable for research in:

- Decentralized machine learning
- Edge AI and IoT systems
- Privacy-preserving learning
- Distributed optimization
- Network communication efficiency
- Model compression and optimization

## Limitations

- **Research Only**: Not suitable for production or safety-critical applications
- **Simulated Environment**: Peer-to-peer communication is simulated
- **Limited Scalability**: Designed for small to medium peer networks
- **Single Task**: Currently supports classification tasks only
- **No Security**: No encryption or security measures implemented

## Contributing

This is a research and educational project. Contributions are welcome for:

- Additional model architectures
- New data distribution strategies
- Enhanced evaluation metrics
- Additional export formats
- Documentation improvements

## License

This project is for research and educational purposes only. Please ensure compliance with your institution's policies and applicable regulations.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{decentralized_ai_systems,
  title={Decentralized AI Systems},
  author={Kryptologyst},
  year={2026},
  note={Research and Educational Use Only}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit team for the interactive demo framework
- The open-source community for various tools and libraries

---

**Remember**: This is a research and educational demonstration only. Do not use in production or safety-critical applications.
# Decentralized-AI-Systems
