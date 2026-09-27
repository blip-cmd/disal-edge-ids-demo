"""
Test: Model Loading and Single-Record Inference

Validates that the self-contained HybridSwinLSTM architecture in model.py
can load both FP32 and INT8 weights and produce correct output shapes.
"""

import os
import sys
import torch

# Add app directory to path
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from model import HybridSwinLSTM


def test_fp32_model_load():
    """Test that FP32 state dict loads into HybridSwinLSTM and produces scalar output."""
    weights_path = os.path.join(APP_DIR, "weights", "transfer_rescue_ciciot_to_botiot_fp32.pth")
    assert os.path.exists(weights_path), f"FP32 weights not found: {weights_path}"

    model = HybridSwinLSTM(input_shape=(41, 1))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Single record inference
    x = torch.randn(1, 41)
    with torch.no_grad():
        out = model(x)

    assert out.shape == torch.Size([1]), f"Expected (1,) output for single record, got shape {out.shape}"
    print(f"[PASS] FP32 model load: output={out.item():.4f}")


def test_fp32_batch_shape():
    """Test that FP32 model handles batch input correctly."""
    weights_path = os.path.join(APP_DIR, "weights", "transfer_rescue_ciciot_to_botiot_fp32.pth")
    model = HybridSwinLSTM(input_shape=(41, 1))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Batch inference
    x = torch.randn(32, 41)
    with torch.no_grad():
        out = model(x)

    assert out.shape == torch.Size([32]), f"Expected (32,) output, got {out.shape}"
    print(f"[PASS] FP32 batch shape: output shape={out.shape}")


def test_int8_model_load():
    """Test that INT8 quantized model loads and produces correct output."""
    model_path = os.path.join(APP_DIR, "weights", "transfer_rescue_ciciot_to_botiot_int8.pth")
    assert os.path.exists(model_path), f"INT8 model not found: {model_path}"

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    x = torch.randn(1, 41)
    with torch.no_grad():
        out = model(x)

    assert out.shape == torch.Size([1]), f"Expected (1,) output for single record, got shape {out.shape}"
    print(f"[PASS] INT8 model load: output={out.item():.4f}")


def test_parameter_count():
    """Verify the model has the expected parameter count."""
    model = HybridSwinLSTM(input_shape=(41, 1))
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 1038383, f"Expected 1,038,383 params, got {total_params:,}"
    print(f"[PASS] Parameter count: {total_params:,}")


def test_sigmoid_output_range():
    """Verify sigmoid probabilities are in [0, 1] range."""
    weights_path = os.path.join(APP_DIR, "weights", "transfer_rescue_ciciot_to_botiot_fp32.pth")
    model = HybridSwinLSTM(input_shape=(41, 1))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    x = torch.randn(100, 41)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    assert probs.min() >= 0.0, f"Probability below 0: {probs.min()}"
    assert probs.max() <= 1.0, f"Probability above 1: {probs.max()}"
    print(f"[PASS] Sigmoid range: min={probs.min():.4f}, max={probs.max():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Model Loading and Inference Tests")
    print("=" * 60)

    test_fp32_model_load()
    test_fp32_batch_shape()
    test_int8_model_load()
    test_parameter_count()
    test_sigmoid_output_range()

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
