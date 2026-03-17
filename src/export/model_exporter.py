"""Model export and deployment utilities for edge AI systems.

This module provides tools for exporting PyTorch models to various edge-optimized
formats including ONNX, TensorFlow Lite, CoreML, and OpenVINO.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Optional imports for different export formats
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available. Install onnx and onnxruntime for ONNX export.")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install tensorflow for TFLite export.")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("CoreML not available. Install coremltools for CoreML export.")

try:
    from openvino.tools import mo
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logging.warning("OpenVINO not available. Install openvino for OpenVINO export.")

logger = logging.getLogger(__name__)


class ModelExporter:
    """Handles model export to various edge-optimized formats.
    
    This class provides utilities for converting PyTorch models to formats
    suitable for deployment on edge devices.
    """
    
    def __init__(self, output_dir: str = "./assets/models") -> None:
        """Initialize the model exporter.
        
        Args:
            output_dir: Directory to save exported models.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        filename: str = "model.onnx",
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> str:
        """Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export.
            input_shape: Input tensor shape (excluding batch dimension).
            filename: Output filename.
            opset_version: ONNX opset version.
            dynamic_axes: Dynamic axes configuration.
            
        Returns:
            Path to exported ONNX model.
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install onnx and onnxruntime.")
        
        logger.info(f"Exporting model to ONNX format: {filename}")
        
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        output_path = self.output_dir / filename
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify the exported model
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model exported successfully: {output_path}")
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            raise
        
        return str(output_path)
    
    def export_to_tflite(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        filename: str = "model.tflite",
        quantize: bool = True,
    ) -> str:
        """Export PyTorch model to TensorFlow Lite format.
        
        Args:
            model: PyTorch model to export.
            input_shape: Input tensor shape (excluding batch dimension).
            filename: Output filename.
            quantize: Whether to apply quantization.
            
        Returns:
            Path to exported TFLite model.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install tensorflow.")
        
        logger.info(f"Exporting model to TFLite format: {filename}")
        
        # First export to ONNX, then convert to TensorFlow
        onnx_path = self.export_to_onnx(
            model, input_shape, filename.replace('.tflite', '.onnx')
        )
        
        # Convert ONNX to TensorFlow
        try:
            import tf2onnx
            tf_model_path = onnx_path.replace('.onnx', '.pb')
            
            # Convert ONNX to TensorFlow
            tf2onnx.convert.from_onnx(onnx_path, tf_model_path)
            
            # Convert TensorFlow to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            output_path = self.output_dir / filename
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite model exported successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
            raise
    
    def export_to_coreml(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        filename: str = "model.mlmodel",
        compute_units: str = "cpu_and_gpu",
    ) -> str:
        """Export PyTorch model to CoreML format.
        
        Args:
            model: PyTorch model to export.
            input_shape: Input tensor shape (excluding batch dimension).
            filename: Output filename.
            compute_units: CoreML compute units ('cpu', 'gpu', 'cpu_and_gpu', 'all').
            
        Returns:
            Path to exported CoreML model.
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreML not available. Install coremltools.")
        
        logger.info(f"Exporting model to CoreML format: {filename}")
        
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to CoreML
        traced_model = torch.jit.trace(model, dummy_input)
        
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
            compute_units=getattr(ct.ComputeUnit, compute_units.upper()),
        )
        
        output_path = self.output_dir / filename
        coreml_model.save(str(output_path))
        
        logger.info(f"CoreML model exported successfully: {output_path}")
        return str(output_path)
    
    def export_to_openvino(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        filename: str = "model.xml",
        precision: str = "FP32",
    ) -> str:
        """Export PyTorch model to OpenVINO format.
        
        Args:
            model: PyTorch model to export.
            input_shape: Input tensor shape (excluding batch dimension).
            filename: Output filename.
            precision: Model precision ('FP32', 'FP16', 'INT8').
            
        Returns:
            Path to exported OpenVINO model.
        """
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO not available. Install openvino.")
        
        logger.info(f"Exporting model to OpenVINO format: {filename}")
        
        # First export to ONNX
        onnx_path = self.export_to_onnx(
            model, input_shape, filename.replace('.xml', '.onnx')
        )
        
        # Convert ONNX to OpenVINO
        try:
            output_path = self.output_dir / filename.replace('.xml', '')
            
            mo.convert_model(
                onnx_path,
                output_dir=str(self.output_dir),
                model_name=output_path.stem,
                compress_to_fp16=(precision == "FP16"),
            )
            
            logger.info(f"OpenVINO model exported successfully: {output_path}")
            return str(output_path.with_suffix('.xml'))
            
        except Exception as e:
            logger.error(f"OpenVINO export failed: {e}")
            raise
    
    def export_all_formats(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        model_name: str = "decentralized_model",
    ) -> Dict[str, str]:
        """Export model to all available formats.
        
        Args:
            model: PyTorch model to export.
            input_shape: Input tensor shape (excluding batch dimension).
            model_name: Base name for exported models.
            
        Returns:
            Dictionary mapping format names to file paths.
        """
        exported_models = {}
        
        # Export to ONNX
        try:
            onnx_path = self.export_to_onnx(
                model, input_shape, f"{model_name}.onnx"
            )
            exported_models['onnx'] = onnx_path
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
        
        # Export to TFLite
        try:
            tflite_path = self.export_to_tflite(
                model, input_shape, f"{model_name}.tflite"
            )
            exported_models['tflite'] = tflite_path
        except Exception as e:
            logger.warning(f"TFLite export failed: {e}")
        
        # Export to CoreML
        try:
            coreml_path = self.export_to_coreml(
                model, input_shape, f"{model_name}.mlmodel"
            )
            exported_models['coreml'] = coreml_path
        except Exception as e:
            logger.warning(f"CoreML export failed: {e}")
        
        # Export to OpenVINO
        try:
            openvino_path = self.export_to_openvino(
                model, input_shape, f"{model_name}.xml"
            )
            exported_models['openvino'] = openvino_path
        except Exception as e:
            logger.warning(f"OpenVINO export failed: {e}")
        
        logger.info(f"Exported models: {list(exported_models.keys())}")
        return exported_models


class ModelOptimizer:
    """Handles model optimization for edge deployment.
    
    This class provides utilities for model compression, quantization,
    and optimization techniques suitable for edge devices.
    """
    
    def __init__(self) -> None:
        """Initialize the model optimizer."""
        pass
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        method: str = "dynamic",
    ) -> nn.Module:
        """Quantize model for reduced precision inference.
        
        Args:
            model: PyTorch model to quantize.
            calibration_data: Data for calibration (for static quantization).
            method: Quantization method ('dynamic', 'static').
            
        Returns:
            Quantized model.
        """
        logger.info(f"Quantizing model using {method} quantization")
        
        if method == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        elif method == "static":
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model_prepared = torch.quantization.prepare(model)
            
            # Calibrate with sample data
            with torch.no_grad():
                for i in range(min(100, len(calibration_data))):
                    model_prepared(calibration_data[i:i+1])
            
            quantized_model = torch.quantization.convert(model_prepared)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        logger.info("Model quantization completed")
        return quantized_model
    
    def prune_model(
        self,
        model: nn.Module,
        sparsity: float = 0.2,
        method: str = "magnitude",
    ) -> nn.Module:
        """Prune model to reduce model size.
        
        Args:
            model: PyTorch model to prune.
            sparsity: Target sparsity ratio.
            method: Pruning method ('magnitude', 'random').
            
        Returns:
            Pruned model.
        """
        logger.info(f"Pruning model with {method} method, sparsity: {sparsity}")
        
        if method == "magnitude":
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    parameters_to_prune.append((module, 'weight'))
            
            torch.nn.utils.prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch.nn.utils.prune.L1Unstructured,
                amount=sparsity,
            )
        elif method == "random":
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    torch.nn.utils.prune.random_unstructured(
                        module, name='weight', amount=sparsity
                    )
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        logger.info("Model pruning completed")
        return model
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: PyTorch model to optimize.
            
        Returns:
            Optimized model.
        """
        logger.info("Optimizing model for inference")
        
        model.eval()
        
        # Fuse operations where possible
        if hasattr(torch.jit, 'fuse'):
            try:
                model = torch.jit.fuse(model)
                logger.info("Model fusion completed")
            except Exception as e:
                logger.warning(f"Model fusion failed: {e}")
        
        # Convert to TorchScript
        try:
            dummy_input = torch.randn(1, 1, 28, 28)
            model = torch.jit.trace(model, dummy_input)
            logger.info("Model converted to TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript conversion failed: {e}")
        
        return model


class EdgeRuntime:
    """Manages edge runtime inference for deployed models.
    
    This class provides utilities for running inference on edge devices
    using various optimized runtimes.
    """
    
    def __init__(self, model_path: str, runtime_type: str = "onnx") -> None:
        """Initialize the edge runtime.
        
        Args:
            model_path: Path to the deployed model.
            runtime_type: Type of runtime ('onnx', 'tflite', 'coreml').
        """
        self.model_path = model_path
        self.runtime_type = runtime_type
        self.session = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model for inference."""
        if self.runtime_type == "onnx" and ONNX_AVAILABLE:
            self.session = ort.InferenceSession(self.model_path)
            logger.info(f"ONNX model loaded: {self.model_path}")
        elif self.runtime_type == "tflite" and TENSORFLOW_AVAILABLE:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            logger.info(f"TFLite model loaded: {self.model_path}")
        else:
            raise ValueError(f"Unsupported runtime type: {self.runtime_type}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data.
        
        Args:
            input_data: Input data array.
            
        Returns:
            Prediction results.
        """
        if self.runtime_type == "onnx":
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: input_data})
            return output[0]
        elif self.runtime_type == "tflite":
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(output_details[0]['index'])
            return output
        else:
            raise ValueError(f"Unsupported runtime type: {self.runtime_type}")
    
    def benchmark_inference(
        self,
        input_data: np.ndarray,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            input_data: Input data for benchmarking.
            num_runs: Number of inference runs.
            
        Returns:
            Dictionary containing performance metrics.
        """
        import time
        
        # Warmup runs
        for _ in range(10):
            self.predict(input_data)
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.predict(input_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean_latency': np.mean(times),
            'std_latency': np.std(times),
            'p50_latency': np.percentile(times, 50),
            'p95_latency': np.percentile(times, 95),
            'p99_latency': np.percentile(times, 99),
            'throughput_fps': 1.0 / np.mean(times),
        }
