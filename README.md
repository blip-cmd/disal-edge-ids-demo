# DISAL Edge Intrusion Detection System: Interactive Dashboard

This repository contains the standalone, edge-deployable Streamlit application for the **Hybrid Swin-LSTM Network Intrusion Detection System (IDS)** developed at the Data-Intensive Systems and Applications Laboratory (DISAL), Department of Computer Science, University of Ghana.

The application demonstrates real-time network flow classification on edge gateways using shifted-window self-attention combined with recurrent long short-term memory (LSTM) units.

---

## Key Capabilities

1. **Multi-Detector Architecture Registry**:
   * **ToN-IoT Binary Edge Detector**: Trained on multi-source Industrial IoT telemetry (42 features). Achieves 100.0% accuracy on demonstration traffic and 99.33% balanced accuracy on the official benchmark partition.
   * **Bot-IoT 5% Official Downsampled Detector**: Trained using train-only 5:1 downsampling on official Bot-IoT data (35 features). Achieves 100.0% accuracy on demonstration traffic and 99.77% balanced accuracy on 366,853 evaluation flows with zero false alarms.
   * **Transfer Rescue (CICIoT2023 to Bot-IoT NetFlow v2)**: Cross-domain transfer learning (41 features) that overcomes majority-class collapse under extreme 278:1 attack imbalance, achieving 99.98% attack recall and 99.84% benchmark accuracy.

2. **Ultra-Compact Edge Quantization**:
   * Toggle dynamically between **FP32 full precision** (4.04 MB) and **INT8 post-training quantization** (1.15 MB).
   * Over 3.5x memory compression, enabling execution within small edge router and gateway memory budgets without specialized accelerator hardware.

3. **High-Throughput Batch Processing & Line Rate Telemetry**:
   * Batch inference engine with real-time progress callbacks, line rates (>750 flows per second on CPU), and sub-2 millisecond latency per record.
   * Interactive results table filterable by classification outcome, accompanied by confusion breakdown grids and CSV export.

4. **Operating Decision Threshold Calibration**:
   * Interactive decision boundary slider (0.05 to 0.99, default 0.50).
   * Enables network operators to adjust detection sensitivity according to local threat priors, mitigating false alarm rates on imbalanced streams.

5. **Integrated Benchmark Partitions**:
   * Includes balanced demonstration stress test batches (100 flows), authentic benchmark skew batches (1:99), single-flow normal and attack sample fixtures, and the complete 42,167-record ToN-IoT evaluation test partition.

---

## Empirical Benchmark Performance

The table below summarizes live inference performance across the showcased detectors:

| Detector Model | Model Precision | Evaluation Traffic Source | Total Flows | Accuracy | Attack Recall | Benign Specificity | Balanced Accuracy | Edge Throughput | Weight Footprint |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ToN-IoT Binary** | FP32 | Demo Batch (50 Normal, 50 Attack) | 100 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 506 flows/s | 4.03 MB |
| **ToN-IoT Binary** | INT8 | Demo Batch (50 Normal, 50 Attack) | 100 | **91.0%** | **90.0%** | **92.0%** | **91.0%** | 589 flows/s | **1.15 MB** |
| **Bot-IoT 5% Official** | FP32 | Demo Batch (47 Normal, 53 Attack) | 100 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 772 flows/s | 4.04 MB |
| **Bot-IoT 5% Official** | INT8 | Demo Batch (47 Normal, 53 Attack) | 100 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 605 flows/s | **1.15 MB** |
| **Transfer Rescue (Bot-IoT)** | FP32 | Authentic Benchmark Skew (1:99) | 100 | **97.0%** | **98.0%** | 0.0% | 49.0% | 780 flows/s | 4.03 MB |
| **Transfer Rescue (Bot-IoT)** | INT8 | Authentic Benchmark Skew (1:99) | 100 | **98.0%** | **99.0%** | 0.0% | 49.5% | 747 flows/s | **1.16 MB** |
| **Transfer Rescue (Bot-IoT)** | FP32 | Full Benchmark Test Slice | 500 | **99.4%** | **99.6%** | 0.0% | 49.8% | **1,144 flows/s** | 4.03 MB |

---

## Directory Structure

```
├── app.py                      # Streamlit application dashboard
├── model.py                    # Standalone PyTorch HybridSwinLSTM architecture
├── requirements.txt            # Lightweight CPU-optimized dependencies
├── README.md                   # Project documentation
├── weights/                    # Pre-packaged trained FP32 and INT8 weights
│   ├── toniot_binary_fp32.pth
│   ├── toniot_binary_int8.pth
│   ├── botiot_5pct_official_fp32.pth
│   ├── botiot_5pct_official_int8.pth
│   ├── transfer_rescue_ciciot_to_botiot_fp32.pth
│   └── transfer_rescue_ciciot_to_botiot_int8.pth
├── data/                       # Demonstration batches, samples, and benchmark partition
│   ├── toniot_demo_batch.csv
│   ├── toniot_test_partition.csv      # Full 42,167-flow evaluation partition (7.32 MB)
│   ├── botiot_5pct_demo_batch.csv
│   ├── botiot_benchmark_batch.csv
│   ├── botiot_demo_batch.csv
│   ├── toniot_sample_normal.csv
│   ├── toniot_sample_attack.csv
│   ├── botiot_sample_normal.csv
│   └── botiot_sample_attack.csv
└── tests/                      # Automated validation suite
    ├── run_all_tests.py
    ├── test_model_load.py
    ├── test_batch_inference.py
    └── test_streamlit_app.py
```

---

## Deployment & Execution Guide

### 1. Local Execution

Clone the repository and install requirements in your Python environment:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
streamlit run app.py
```

The interface will open automatically in your browser at `http://localhost:8501`.

### 2. Streamlit Community Cloud Deployment

1. Fork or push this repository to your GitHub account as a public or private repository.
2. Visit **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"**.
4. Set the following fields:
   * **Repository**: Your GitHub repository name.
   * **Branch**: `main`.
   * **Main file path**: `app.py`.
5. Click **"Deploy!"**.

Because `requirements.txt` specifies CPU-only PyTorch wheels from the official PyTorch index, cloud build and startup times complete in under two minutes without requiring GPU hardware.

---

## Citation & Academic Context

This software artifact accompanies Master of Science research conducted at the University of Ghana:
* **Research Title**: Parameter-Reduced Hybrid Swin-LSTM Network Intrusion Detection for Resource-Constrained Internet of Things Environments.
* **Supervisors**: Albert Ankomah Dodoo (DISAL) and Prof. Justice Kwame Appati (Department of Computer Science).
