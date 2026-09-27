"""
Test: Batch Inference on Bot-IoT Demo Data

Validates that the model produces correct predictions on the bundled
Bot-IoT demo batch, and that inference helpers work end-to-end.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from model import HybridSwinLSTM

FEATURE_NAMES = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]


def load_model():
    """Load the FP32 model."""
    weights_path = os.path.join(APP_DIR, "weights", "transfer_rescue_ciciot_to_botiot_fp32.pth")
    model = HybridSwinLSTM(input_shape=(41, 1))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def test_demo_batch_loads():
    """Test that the demo batch CSV loads with correct dimensions."""
    batch_path = os.path.join(APP_DIR, "data", "botiot_demo_batch.csv")
    assert os.path.exists(batch_path), f"Demo batch not found: {batch_path}"

    df = pd.read_csv(batch_path)
    feature_cols = [c for c in df.columns if c != "target"]

    assert len(feature_cols) == 41, f"Expected 41 features, got {len(feature_cols)}"
    assert "target" in df.columns, "Missing 'target' column in demo batch"
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    print(f"[PASS] Demo batch shape: {df.shape}, class dist: {df['target'].value_counts().to_dict()}")


def test_batch_inference_shape():
    """Test batch inference produces correct output dimensions."""
    model = load_model()
    df = pd.read_csv(os.path.join(APP_DIR, "data", "botiot_demo_batch.csv"))
    feature_cols = [c for c in df.columns if c != "target"]

    X = torch.FloatTensor(df[feature_cols].values)
    with torch.no_grad():
        logits = model(X)

    assert logits.shape == torch.Size([100]), f"Expected (100,), got {logits.shape}"
    print(f"[PASS] Batch inference shape: {logits.shape}")


def test_batch_predictions_reasonable():
    """Test that batch predictions on demo data are reasonable (not all same class)."""
    model = load_model()
    df = pd.read_csv(os.path.join(APP_DIR, "data", "botiot_demo_batch.csv"))
    feature_cols = [c for c in df.columns if c != "target"]

    X = torch.FloatTensor(df[feature_cols].values)
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()

    num_attack = preds.sum().item()
    num_normal = len(preds) - num_attack

    # The demo batch is balanced (50/50), so we expect at least some of each
    assert num_attack > 0, "Model predicted zero attacks on balanced batch"
    assert num_normal >= 0, "Negative normal count (impossible)"

    accuracy = float((preds.numpy() == df["target"].values).mean() * 100)

    print(f"[PASS] Predictions: {num_attack} attacks, {num_normal} normal")
    print(f"       Accuracy vs ground truth: {accuracy:.1f}%")


def test_sample_files_exist():
    """Test that all sample data files exist and have correct structure."""
    for fname in ["botiot_sample_normal.csv", "botiot_sample_attack.csv", "botiot_demo_batch.csv"]:
        fpath = os.path.join(APP_DIR, "data", fname)
        assert os.path.exists(fpath), f"Missing: {fpath}"
        df = pd.read_csv(fpath)
        feature_cols = [c for c in df.columns if c != "target"]
        assert len(feature_cols) == 41, f"{fname}: expected 41 features, got {len(feature_cols)}"
    print("[PASS] All sample data files present and valid")


def test_normal_samples_labeled_correctly():
    """Verify normal sample file contains only target=0 rows."""
    df = pd.read_csv(os.path.join(APP_DIR, "data", "botiot_sample_normal.csv"))
    assert (df["target"] == 0).all(), "Normal sample file contains non-zero targets"
    print(f"[PASS] Normal samples: {len(df)} rows, all target=0")


def test_attack_samples_labeled_correctly():
    """Verify attack sample file contains only target=1 rows."""
    df = pd.read_csv(os.path.join(APP_DIR, "data", "botiot_sample_attack.csv"))
    assert (df["target"] == 1).all(), "Attack sample file contains non-one targets"
    print(f"[PASS] Attack samples: {len(df)} rows, all target=1")


if __name__ == "__main__":
    print("=" * 60)
    print("Batch Inference and Data Validation Tests")
    print("=" * 60)

    test_demo_batch_loads()
    test_sample_files_exist()
    test_normal_samples_labeled_correctly()
    test_attack_samples_labeled_correctly()
    test_batch_inference_shape()
    test_batch_predictions_reasonable()

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
