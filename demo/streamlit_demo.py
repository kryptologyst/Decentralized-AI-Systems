"""Streamlit demo for Decentralized AI Systems.

This interactive demo simulates decentralized learning on edge devices
with real-time visualization of training progress and metrics.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import json
from pathlib import Path

# Import our modules
from src.models.decentralized_learning import (
    EdgeCNN, DecentralizedPeer, DecentralizedNetwork, set_deterministic_seed
)
from src.pipelines.data_handler import DataDistributor
from src.utils.evaluator import DecentralizedEvaluator

# Page configuration
st.set_page_config(
    page_title="Decentralized AI Systems Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 Decentralized AI Systems Demo</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p><strong>This is a research and educational demonstration only.</strong> 
    This system is NOT intended for safety-critical applications or production use. 
    The models and algorithms shown here are for research and educational purposes only.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Simulation parameters
st.sidebar.subheader("Simulation Parameters")
num_peers = st.sidebar.slider("Number of Peers", 2, 8, 3)
total_rounds = st.sidebar.slider("Training Rounds", 5, 20, 10)
local_epochs = st.sidebar.slider("Local Epochs per Round", 1, 5, 1)
sync_frequency = st.sidebar.slider("Sync Frequency", 1, 5, 1)

# Data distribution
st.sidebar.subheader("Data Distribution")
distribution_strategy = st.sidebar.selectbox(
    "Distribution Strategy",
    ["iid", "non_iid", "heterogeneous"],
    index=1
)

use_synthetic = st.sidebar.checkbox("Use Synthetic Data", value=True)
if use_synthetic:
    num_samples = st.sidebar.slider("Number of Samples", 1000, 10000, 5000)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1)

# Model parameters
st.sidebar.subheader("Model Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

# Device selection
st.sidebar.subheader("Device Configuration")
device_type = st.sidebar.selectbox("Device Type", ["CPU", "CUDA (if available)"], index=0)
device = torch.device("cuda" if device_type == "CUDA (if available)" and torch.cuda.is_available() else "cpu")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Training Simulation", "Model Performance", "Communication Analysis", "Edge Deployment"])

with tab1:
    st.header("Decentralized Training Simulation")
    
    if st.button("🚀 Start Training Simulation", type="primary"):
        # Set deterministic seed
        set_deterministic_seed(42)
        
        # Initialize data distributor
        data_distributor = DataDistributor(distribution_strategy=distribution_strategy)
        
        # Load data
        with st.spinner("Loading and distributing data..."):
            if use_synthetic:
                x_train, y_train, x_test, y_test = data_distributor.load_synthetic_data(
                    num_samples=num_samples,
                    num_classes=10,
                    input_shape=(1, 28, 28),
                    noise_level=noise_level
                )
            else:
                x_train, y_train, x_test, y_test = data_distributor.load_mnist_data()
        
        # Distribute data to peers
        peer_data = data_distributor.distribute_data_to_peers(
            x_train, y_train, num_peers=num_peers
        )
        
        # Create data loaders
        peer_dataloaders = data_distributor.create_dataloaders(
            peer_data, batch_size=batch_size, shuffle=True
        )
        
        test_dataset = TensorDataset(x_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create peers
        peers = []
        for peer_id in range(num_peers):
            model = EdgeCNN(num_classes=10, input_channels=1)
            peer = DecentralizedPeer(
                peer_id=peer_id,
                model=model,
                device=device,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            peers.append(peer)
        
        # Create network
        network = DecentralizedNetwork(
            peers=peers,
            sync_frequency=sync_frequency,
            sync_strategy="average"
        )
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Metrics tracking
        round_accuracies = []
        round_losses = []
        communication_overhead = []
        
        # Training loop
        for round_num in range(total_rounds):
            status_text.text(f"Training Round {round_num + 1}/{total_rounds}")
            
            # Local training phase
            for peer, dataloader in zip(peers, peer_dataloaders):
                peer.train_local(dataloader, epochs=local_epochs)
            
            # Synchronization phase
            if (round_num + 1) % sync_frequency == 0:
                network.synchronize_models()
            
            # Evaluation phase
            round_losses_peer = []
            round_accuracies_peer = []
            
            for peer in peers:
                loss, accuracy = peer.evaluate(test_dataloader)
                round_losses_peer.append(loss)
                round_accuracies_peer.append(accuracy)
            
            avg_loss = np.mean(round_losses_peer)
            avg_accuracy = np.mean(round_accuracies_peer)
            
            round_losses.append(avg_loss)
            round_accuracies.append(avg_accuracy)
            communication_overhead.append(len(peers) * (len(peers) - 1))
            
            # Update progress
            progress_bar.progress((round_num + 1) / total_rounds)
            
            # Small delay for visualization
            time.sleep(0.1)
        
        status_text.text("Training completed!")
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Accuracy", f"{round_accuracies[-1]:.2f}%")
        
        with col2:
            st.metric("Final Loss", f"{round_losses[-1]:.4f}")
        
        with col3:
            st.metric("Total Rounds", total_rounds)
        
        with col4:
            st.metric("Total Peers", num_peers)
        
        # Training progress plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Accuracy Progress", "Loss Progress", 
                          "Communication Overhead", "Peer Comparison"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        rounds = list(range(1, total_rounds + 1))
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=rounds, y=round_accuracies, mode='lines+markers', 
                      name='Network Accuracy', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=rounds, y=round_losses, mode='lines+markers', 
                      name='Network Loss', line=dict(color='red')),
            row=1, col=2
        )
        
        # Communication overhead
        fig.add_trace(
            go.Scatter(x=rounds, y=communication_overhead, mode='lines+markers', 
                      name='Comm Overhead', line=dict(color='green')),
            row=2, col=1
        )
        
        # Peer comparison (final accuracies)
        peer_final_accuracies = [peer.training_accuracies[-1] if peer.training_accuracies else 0 
                               for peer in peers]
        fig.add_trace(
            go.Bar(x=[f"Peer {i}" for i in range(num_peers)], y=peer_final_accuracies,
                   name='Peer Accuracies', marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Training Progress Analysis")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Performance Analysis")
    
    if 'peers' in locals() and peers:
        # Model evaluation
        evaluator = DecentralizedEvaluator()
        
        # Get representative model (first peer)
        representative_model = peers[0].model
        
        # Evaluate model quality
        quality_metrics = evaluator.evaluate_model_quality(
            model=representative_model,
            test_dataloader=test_dataloader,
            device=device
        )
        
        # Display quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{quality_metrics['accuracy']:.4f}")
        
        with col2:
            st.metric("Precision", f"{quality_metrics['precision']:.4f}")
        
        with col3:
            st.metric("Recall", f"{quality_metrics['recall']:.4f}")
        
        with col4:
            st.metric("F1-Score", f"{quality_metrics['f1_score']:.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = quality_metrics['confusion_matrix']
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Per-class metrics
        st.subheader("Per-Class Performance")
        class_metrics_df = pd.DataFrame({
            'Class': range(10),
            'Precision': quality_metrics['precision_per_class'],
            'Recall': quality_metrics['recall_per_class'],
            'F1-Score': quality_metrics['f1_per_class']
        })
        
        st.dataframe(class_metrics_df, use_container_width=True)
        
        # Model size and efficiency
        st.subheader("Model Efficiency")
        
        # Calculate model size
        total_params = sum(p.numel() for p in representative_model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in representative_model.parameters()) / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Parameters", f"{total_params:,}")
        
        with col2:
            st.metric("Model Size", f"{model_size_mb:.2f} MB")
        
        with col3:
            st.metric("Device", str(device))
    
    else:
        st.info("Please run the training simulation first to see model performance metrics.")

with tab3:
    st.header("Communication Analysis")
    
    if 'network' in locals() and network:
        # Get network statistics
        network_stats = network.get_network_statistics()
        
        # Communication metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Peers", network_stats['total_peers'])
        
        with col2:
            st.metric("Communication Rounds", network_stats['total_communication_rounds'])
        
        with col3:
            st.metric("Sync Operations", network_stats['total_communication_overhead'])
        
        with col4:
            st.metric("Sync Strategy", network_stats['sync_strategy'])
        
        # Communication efficiency analysis
        st.subheader("Communication Efficiency")
        
        # Calculate communication metrics
        model_size_mb = 2.5  # Approximate model size
        total_data_mb = network_stats['total_communication_overhead'] * model_size_mb
        avg_data_per_round = total_data_mb / network_stats['total_communication_rounds']
        
        comm_metrics_df = pd.DataFrame({
            'Metric': ['Total Data Transferred (MB)', 'Avg Data per Round (MB)', 
                      'Avg Data per Peer (MB)', 'Sync Operations'],
            'Value': [f"{total_data_mb:.2f}", f"{avg_data_per_round:.2f}", 
                     f"{total_data_mb/network_stats['total_peers']:.2f}", 
                     network_stats['total_communication_overhead']]
        })
        
        st.dataframe(comm_metrics_df, use_container_width=True)
        
        # Bandwidth requirements
        st.subheader("Bandwidth Requirements")
        
        network_speeds = {
            'WiFi 6': 1200,  # Mbps
            'WiFi 5': 433,   # Mbps
            '4G': 100,       # Mbps
            '3G': 3,         # Mbps
        }
        
        bandwidth_df = pd.DataFrame({
            'Network Type': list(network_speeds.keys()),
            'Speed (Mbps)': list(network_speeds.values()),
            'Sync Time (seconds)': [
                avg_data_per_round * 8 / speed for speed in network_speeds.values()
            ],
            'Feasible': [
                avg_data_per_round * 8 / speed < 60 for speed in network_speeds.values()
            ]
        })
        
        st.dataframe(bandwidth_df, use_container_width=True)
        
        # Communication visualization
        st.subheader("Communication Pattern")
        
        # Simulate communication pattern
        rounds = list(range(1, network_stats['total_communication_rounds'] + 1))
        comm_data = [avg_data_per_round] * len(rounds)
        
        fig_comm = go.Figure()
        fig_comm.add_trace(go.Scatter(
            x=rounds, 
            y=comm_data, 
            mode='lines+markers',
            name='Data per Round',
            line=dict(color='purple', width=3)
        ))
        
        fig_comm.update_layout(
            title="Communication Pattern Over Time",
            xaxis_title="Training Round",
            yaxis_title="Data Transferred (MB)",
            height=400
        )
        
        st.plotly_chart(fig_comm, use_container_width=True)
    
    else:
        st.info("Please run the training simulation first to see communication analysis.")

with tab4:
    st.header("Edge Deployment")
    
    st.subheader("Model Export Options")
    
    # Export formats
    export_formats = st.multiselect(
        "Select Export Formats",
        ["ONNX", "TensorFlow Lite", "CoreML", "OpenVINO"],
        default=["ONNX", "TensorFlow Lite"]
    )
    
    # Device targets
    st.subheader("Target Devices")
    
    device_configs = {
        "Raspberry Pi 4": {
            "memory": "4GB",
            "cpu": "4 cores",
            "recommended_format": "ONNX",
            "max_latency": "100ms"
        },
        "NVIDIA Jetson Nano": {
            "memory": "4GB", 
            "cpu": "4 cores",
            "recommended_format": "TensorRT",
            "max_latency": "50ms"
        },
        "Android Device": {
            "memory": "2-8GB",
            "cpu": "ARM",
            "recommended_format": "TensorFlow Lite",
            "max_latency": "200ms"
        },
        "iOS Device": {
            "memory": "2-8GB",
            "cpu": "ARM",
            "recommended_format": "CoreML",
            "max_latency": "100ms"
        }
    }
    
    selected_device = st.selectbox("Select Target Device", list(device_configs.keys()))
    
    # Display device info
    device_info = device_configs[selected_device]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Memory", device_info["memory"])
    
    with col2:
        st.metric("CPU", device_info["cpu"])
    
    with col3:
        st.metric("Recommended Format", device_info["recommended_format"])
    
    with col4:
        st.metric("Max Latency", device_info["max_latency"])
    
    # Deployment simulation
    st.subheader("Deployment Simulation")
    
    if st.button("🚀 Simulate Edge Deployment"):
        with st.spinner("Simulating edge deployment..."):
            # Simulate deployment process
            progress_bar = st.progress(0)
            
            steps = [
                "Model Optimization",
                "Format Conversion", 
                "Device Validation",
                "Performance Testing",
                "Deployment Complete"
            ]
            
            for i, step in enumerate(steps):
                time.sleep(0.5)
                progress_bar.progress((i + 1) / len(steps))
                st.text(f"Step {i+1}: {step}")
            
            # Simulated performance metrics
            st.success("Deployment simulation completed!")
            
            # Display simulated performance
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Inference Latency", "45ms", "5ms")
            
            with col2:
                st.metric("Memory Usage", "180MB", "20MB")
            
            with col3:
                st.metric("Power Consumption", "2.1W", "0.3W")
    
    # Edge deployment best practices
    st.subheader("Edge Deployment Best Practices")
    
    best_practices = [
        "Use quantization to reduce model size and improve inference speed",
        "Implement model pruning to remove unnecessary parameters",
        "Choose appropriate batch size (usually 1 for edge devices)",
        "Use hardware-specific optimizations (TensorRT, CoreML, etc.)",
        "Implement proper error handling and fallback mechanisms",
        "Monitor resource usage and implement throttling if needed",
        "Use efficient data preprocessing pipelines",
        "Implement model versioning and OTA update capabilities"
    ]
    
    for i, practice in enumerate(best_practices, 1):
        st.write(f"{i}. {practice}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Decentralized AI Systems Demo - Research & Educational Use Only</p>
    <p>Built with PyTorch, Streamlit, and Plotly</p>
</div>
""", unsafe_allow_html=True)
