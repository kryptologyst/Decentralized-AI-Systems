"""Utility functions and helpers for decentralized AI systems."""

import logging
import time
import psutil
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import yaml

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """Get system information for edge device profiling.
    
    Returns:
        Dictionary containing system information.
    """
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_usage_gb': psutil.disk_usage('/').total / (1024**3),
        'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
    }
    
    # GPU information if available
    if torch.cuda.is_available():
        info.update({
            'gpu_available': True,
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'gpu_memory_gb': [torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                            for i in range(torch.cuda.device_count())],
        })
    else:
        info['gpu_available'] = False
    
    return info


def measure_inference_time(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """Measure inference time for a model.
    
    Args:
        model: PyTorch model to measure.
        input_tensor: Input tensor for inference.
        num_runs: Number of inference runs for measurement.
        warmup_runs: Number of warmup runs.
        
    Returns:
        Dictionary containing timing statistics.
    """
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Measure inference times
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'p50_time': np.percentile(times, 50),
        'p95_time': np.percentile(times, 95),
        'p99_time': np.percentile(times, 99),
    }


def calculate_model_complexity(model: nn.Module) -> Dict[str, Any]:
    """Calculate model complexity metrics.
    
    Args:
        model: PyTorch model to analyze.
        
    Returns:
        Dictionary containing complexity metrics.
    """
    total_params = 0
    trainable_params = 0
    total_size = 0
    
    for name, param in model.named_parameters():
        param_size = param.numel() * param.element_size()
        total_size += param_size
        total_params += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
    
    # Count layers by type
    layer_counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in ['Sequential', 'Module']:
            layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_size / (1024 * 1024),
        'layer_counts': layer_counts,
        'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0,
    }


def save_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = 'json'
) -> None:
    """Save results to file.
    
    Args:
        results: Results dictionary to save.
        output_path: Path to save the results.
        format: File format ('json' or 'yaml').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format.lower() == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to: {output_path}")


def load_results(
    input_path: Union[str, Path],
    format: str = 'json'
) -> Dict[str, Any]:
    """Load results from file.
    
    Args:
        input_path: Path to load results from.
        format: File format ('json' or 'yaml').
        
    Returns:
        Loaded results dictionary.
    """
    input_path = Path(input_path)
    
    if format.lower() == 'json':
        with open(input_path, 'r') as f:
            return json.load(f)
    elif format.lower() == 'yaml':
        with open(input_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_experiment_log(
    experiment_name: str,
    config: Dict[str, Any],
    output_dir: Union[str, Path]
) -> Path:
    """Create an experiment log directory and save configuration.
    
    Args:
        experiment_name: Name of the experiment.
        config: Experiment configuration.
        output_dir: Output directory for the experiment.
        
    Returns:
        Path to the experiment directory.
    """
    output_dir = Path(output_dir)
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = experiment_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create subdirectories
    (experiment_dir / 'models').mkdir(exist_ok=True)
    (experiment_dir / 'results').mkdir(exist_ok=True)
    (experiment_dir / 'plots').mkdir(exist_ok=True)
    (experiment_dir / 'logs').mkdir(exist_ok=True)
    
    logger.info(f"Experiment log created: {experiment_dir}")
    return experiment_dir


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_size(bytes_size: int) -> str:
    """Format size in bytes to human-readable format.
    
    Args:
        bytes_size: Size in bytes.
        
    Returns:
        Formatted size string.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate.
        required_keys: List of required keys.
        
    Returns:
        True if configuration is valid, False otherwise.
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        return False
    
    return True


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level.
        log_file: Optional log file path.
        log_format: Optional custom log format.
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available.
    
    Returns:
        Dictionary mapping package names to availability.
    """
    dependencies = {
        'torch': False,
        'torchvision': False,
        'numpy': False,
        'pandas': False,
        'sklearn': False,
        'matplotlib': False,
        'plotly': False,
        'streamlit': False,
        'onnx': False,
        'onnxruntime': False,
        'tensorflow': False,
        'coremltools': False,
        'openvino': False,
    }
    
    for package in dependencies:
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    return dependencies


def print_system_summary() -> None:
    """Print a summary of the system and available dependencies."""
    print("=" * 60)
    print("DECENTRALIZED AI SYSTEMS - SYSTEM SUMMARY")
    print("=" * 60)
    
    # System info
    system_info = get_system_info()
    print(f"CPU Cores: {system_info['cpu_count']}")
    print(f"Memory: {system_info['memory_total_gb']:.1f} GB")
    print(f"Python Version: {system_info['python_version']}")
    
    if system_info['gpu_available']:
        print(f"GPU Available: Yes ({system_info['gpu_count']} devices)")
        for i, name in enumerate(system_info['gpu_names']):
            memory = system_info['gpu_memory_gb'][i]
            print(f"  GPU {i}: {name} ({memory:.1f} GB)")
    else:
        print("GPU Available: No")
    
    print("\nDependencies:")
    deps = check_dependencies()
    for package, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {package}")
    
    print("=" * 60)
