"""Benchmark Showcase Runner
Executes the live application inference pipeline across the showcased detectors:
1. ToN-IoT Binary Edge Detector (FP32 & INT8)
2. Bot-IoT 5% Official Downsampled Detector (FP32 & INT8)
3. Bot-IoT NetFlow v2 Transfer Rescue (Benchmark Batch, Balanced Batch @ 0.50, and Balanced Batch @ Tuned Threshold)
4. Bot-IoT NetFlow v2 on Full Evaluation Test Partition slice (500 flows)
"""

import sys
import time
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch

# Ensure hybrid-model-app is on python path
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from model import HybridSwinLSTM
from app import AVAILABLE_DETECTORS, get_model, run_batch_inference

def run_experiment(detector_name, precision, batch_file_path, threshold=0.5, desc=""):
    detector_info = AVAILABLE_DETECTORS[detector_name]
    model = get_model(precision, detector_info)
    df = pd.read_csv(batch_file_path)
    
    # Handle possible extra target columns if test partition is used
    if "multi_label" in df.columns and "target" in df.columns:
        df = df.drop(columns=["multi_label"])
    elif "multi_label" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"multi_label": "target"})
        
    results_df, summary = run_batch_inference(model, df, threshold=threshold)
    
    tp = summary["tp"] or 0
    tn = summary["tn"] or 0
    fp = summary["fp"] or 0
    fn = summary["fn"] or 0
    total = summary["total"]
    
    normal_total = tn + fp
    attack_total = tp + fn
    
    specificity = (tn / normal_total * 100.0) if normal_total > 0 else 0.0
    recall = (tp / attack_total * 100.0) if attack_total > 0 else 0.0
    precision_rate = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    far = (fp / normal_total * 100.0) if normal_total > 0 else 0.0
    balanced_acc = (recall + specificity) / 2.0
    
    size_mb = detector_info["int8_size_mb"] if "INT8" in precision else detector_info["fp32_size_mb"]
    
    return {
        "detector": detector_name,
        "precision": precision,
        "description": desc,
        "batch_file": Path(batch_file_path).name,
        "records": total,
        "normal_true": normal_total,
        "attack_true": attack_total,
        "threshold": threshold,
        "accuracy": summary["accuracy"],
        "recall": recall,
        "specificity": specificity,
        "precision_rate": precision_rate,
        "far": far,
        "f1": summary["f1"],
        "balanced_acc": balanced_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "throughput_fps": summary["throughput_fps"],
        "latency_per_record_ms": summary["latency_per_record_ms"],
        "size_mb": size_mb,
    }

def main():
    print("=" * 80)
    print("RUNNING LIVE APPLICATION BENCHMARK SHOWCASE")
    print("=" * 80)
    
    runs = [
        # Tier 1: ToN-IoT
        ("ToN-IoT Binary Edge Detector (From Scratch)", "FP32 (4.04 MB)", APP_DIR / "data/toniot_demo_batch.csv", 0.50, "Demo Batch (50 Normal, 50 Attack)"),
        ("ToN-IoT Binary Edge Detector (From Scratch)", "INT8 (1.15 MB)", APP_DIR / "data/toniot_demo_batch.csv", 0.50, "Demo Batch (50 Normal, 50 Attack)"),
        
        # Tier 1: Bot-IoT 5% Official
        ("Bot-IoT 5% Official Downsampled Detector", "FP32 (4.04 MB)", APP_DIR / "data/botiot_5pct_demo_batch.csv", 0.50, "Demo Batch (47 Normal, 53 Attack)"),
        ("Bot-IoT 5% Official Downsampled Detector", "INT8 (1.15 MB)", APP_DIR / "data/botiot_5pct_demo_batch.csv", 0.50, "Demo Batch (47 Normal, 53 Attack)"),
        
        # Tier 3: Bot-IoT NetFlow v2 Transfer Rescue
        ("Transfer Rescue: CICIoT2023 -> Bot-IoT (NetFlow v2)", "FP32 (4.04 MB)", APP_DIR / "data/botiot_benchmark_batch.csv", 0.50, "Benchmark Skew (1 Normal, 99 Attack)"),
        ("Transfer Rescue: CICIoT2023 -> Bot-IoT (NetFlow v2)", "INT8 (1.15 MB)", APP_DIR / "data/botiot_benchmark_batch.csv", 0.50, "Benchmark Skew (1 Normal, 99 Attack)"),
        ("Transfer Rescue: CICIoT2023 -> Bot-IoT (NetFlow v2)", "FP32 (4.04 MB)", APP_DIR / "data/botiot_demo_batch.csv", 0.50, "Balanced Stress Test @ default tau=0.50"),
        ("Transfer Rescue: CICIoT2023 -> Bot-IoT (NetFlow v2)", "FP32 (4.04 MB)", APP_DIR / "data/botiot_demo_batch.csv", 0.99, "Balanced Stress Test @ tuned tau=0.99 (Slider)"),
    ]
    
    # Check if full partition slice exists
    tl_key = "Transfer Rescue: CICIoT2023 -> Bot-IoT (NetFlow v2)"
    full_botiot_test = Path(AVAILABLE_DETECTORS[tl_key]["full_test_path"])
    if full_botiot_test.exists():
        temp_slice = APP_DIR / "data" / "_temp_botiot_slice_500.csv"
        df_full = pd.read_csv(full_botiot_test, nrows=500)
        df_full.to_csv(temp_slice, index=False)
        runs.append(
            (tl_key, "FP32 (4.04 MB)", temp_slice, 0.50, "Full Test Partition Slice (500 flows, 498 Attack, 2 Normal)")
        )
        
    results = []
    for r in runs:
        res = run_experiment(*r)
        results.append(res)
        print(f"[DONE] {res['detector'][:25]} | {res['precision'][:4]} | {res['description'][:35]} | Acc: {res['accuracy']:.1f}% | BalAcc: {res['balanced_acc']:.1f}%")
        
    # Clean up temp slice if created
    temp_slice = APP_DIR / "data" / "_temp_botiot_slice_500.csv"
    if temp_slice.exists():
        temp_slice.unlink()
        
    output_json = APP_DIR / "tests" / "benchmark_showcase_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 80)
    print(f"Results saved to {output_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
