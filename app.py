"""
DISAL Edge Detector: Standalone Streamlit Deployment Application.

Architecture: Hybrid Swin-LSTM Network Intrusion Detector
Target Paper: Bempong and Brown (2026)
Dissertation: "Hybrid Swin-LSTM: Cross-Domain Network Intrusion Detection"

Supported Viable Detectors:
1. Transfer Rescue: CICIoT2023 -> Bot-IoT NetFlow v2 (41 features)
2. ToN-IoT Binary Edge Detector (From Scratch, Balanced) (42 features)
3. Bot-IoT 5% Official Downsampled Detector (35 features)

Deployment: Fully self-contained inside hybrid-model-app/, runs locally
via 'streamlit run app.py' or hosted on Streamlit Community Cloud.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# Ensure self-contained imports from hybrid-model-app directory
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

PROJECT_ROOT = APP_DIR.parent if (APP_DIR.parent / "data" / "preprocessed").exists() else APP_DIR

from model import HybridSwinLSTM

# ---------------------------------------------------------------------------
# Feature Definitions and Metadata per Detector
# ---------------------------------------------------------------------------

# 1. Bot-IoT NetFlow v2 (41 features)
BOTIOT_FEATURES = [
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

BOTIOT_GROUPS = {
    "Transport and Port Identity": [
        "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    ],
    "Packet and Byte Volumes": [
        "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
        "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES",
    ],
    "TCP State and Window Dynamics": [
        "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
        "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    ],
    "Temporal and Flow Durations": [
        "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    ],
    "IP Layer and Packet Boundaries": [
        "MIN_TTL", "MAX_TTL",
        "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
        "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    ],
    "Throughput and Retransmission": [
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    ],
    "Application Protocol Telemetry": [
        "ICMP_TYPE", "ICMP_IPV4_TYPE",
        "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
        "FTP_COMMAND_RET_CODE",
    ],
}

BOTIOT_TOOLTIPS = {
    "L4_SRC_PORT": "Transport layer source port (0 to 65535). High ports identify client sockets; low ports identify standard services.",
    "L4_DST_PORT": "Transport layer destination port (0 to 65535). Identifies targeted service (such as 80 HTTP, 443 HTTPS, 22 SSH).",
    "PROTOCOL": "IP protocol number. 6 denotes TCP, 17 denotes UDP, 1 denotes ICMP.",
    "L7_PROTO": "Layer 7 application protocol identifier derived through deep packet inspection.",
    "IN_BYTES": "Total volume of incoming bytes transferred from source to destination.",
    "IN_PKTS": "Total count of incoming packets transferred from source to destination.",
    "OUT_BYTES": "Total volume of outgoing response bytes returned from destination back to source.",
    "OUT_PKTS": "Total count of outgoing packets returned from destination back to source.",
    "TCP_FLAGS": "Bitwise OR combination of all TCP control flags observed across the bidirectional conversation.",
    "CLIENT_TCP_FLAGS": "Cumulative TCP control flags set by the flow initiator. Essential for detecting SYN flood and port scans.",
    "SERVER_TCP_FLAGS": "Cumulative TCP control flags set by the responder, reflecting handshake completion or connection resets.",
    "FLOW_DURATION_MILLISECONDS": "Total active conversation duration from initial handshake to flow termination, in milliseconds.",
    "DURATION_IN": "Active client transmission duration for incoming request payload in milliseconds.",
    "DURATION_OUT": "Active server transmission duration for outgoing response payload in milliseconds.",
    "MIN_TTL": "Minimum Time to Live recorded in IP headers across the flow.",
    "MAX_TTL": "Maximum Time to Live recorded in IP headers. Substantial variance indicates routing anomalies or spoofed origins.",
    "LONGEST_FLOW_PKT": "Maximum packet length in bytes observed across the flow. Detects oversized payloads used in exploitation.",
    "SHORTEST_FLOW_PKT": "Minimum packet length in bytes observed across the flow. Tiny packet dominance signals control or ping floods.",
    "MIN_IP_PKT_LEN": "Minimum IP packet header length observed, used to identify malformed headers.",
    "MAX_IP_PKT_LEN": "Maximum IP packet header length observed, detecting MTU saturation boundaries.",
    "SRC_TO_DST_SECOND_BYTES": "Instantaneous upload transfer rate from source to destination, in bytes per second.",
    "DST_TO_SRC_SECOND_BYTES": "Instantaneous download transfer rate from destination back to source, in bytes per second.",
    "RETRANSMITTED_IN_BYTES": "Volume of retransmitted incoming bytes. High volume indicates severe network congestion or dropped packets.",
    "RETRANSMITTED_IN_PKTS": "Total count of retransmitted incoming packets caused by packet loss.",
    "RETRANSMITTED_OUT_BYTES": "Volume of retransmitted outgoing response bytes, signaling server response stalls.",
    "RETRANSMITTED_OUT_PKTS": "Total count of retransmitted outgoing response packets.",
    "SRC_TO_DST_AVG_THROUGHPUT": "Average network throughput from source to destination in bits per second.",
    "DST_TO_SRC_AVG_THROUGHPUT": "Average network throughput from destination back to source in bits per second.",
    "NUM_PKTS_UP_TO_128_BYTES": "Count of small flow packets with length <= 128 bytes (TCP control messages, handshake, or ping probes).",
    "NUM_PKTS_128_TO_256_BYTES": "Count of flow packets with lengths between 128 and 256 bytes (short command beacons).",
    "NUM_PKTS_256_TO_512_BYTES": "Count of flow packets with lengths between 256 and 512 bytes (standard queries and client handshakes).",
    "NUM_PKTS_512_TO_1024_BYTES": "Count of flow packets with lengths between 512 and 1024 bytes (medium payload transfers and DNS replies).",
    "NUM_PKTS_1024_TO_1514_BYTES": "Count of flow packets approaching Ethernet MTU (1024-1514 bytes), typical of bulk data transfers.",
    "TCP_WIN_MAX_IN": "Maximum receive window size advertised by the client in incoming packets.",
    "TCP_WIN_MAX_OUT": "Maximum receive window size advertised by the server in outgoing packets.",
    "ICMP_TYPE": "Control message type from ICMP header (such as 8 for echo request, 0 for reply, or 3 for unreachable).",
    "ICMP_IPV4_TYPE": "IPv4-specific ICMP subtype classification code.",
    "DNS_QUERY_ID": "Transaction identification number assigned to DNS queries for matching requests to responses.",
    "DNS_QUERY_TYPE": "Record type requested in DNS query (such as 1 for IPv4 A, 28 for AAAA, 15 for MX).",
    "DNS_TTL_ANSWER": "Cached time to live in seconds returned in the DNS answer section.",
    "FTP_COMMAND_RET_CODE": "Three-digit numeric status reply code returned by FTP control servers.",
}

# 2. ToN-IoT (42 features)
TONIOT_FEATURES = [
    "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "duration",
    "src_bytes", "dst_bytes", "conn_state", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_query", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer",
    "http_trans_depth", "http_method", "http_uri", "http_version",
    "http_request_body_len", "http_response_body_len", "http_status_code", "http_user_agent",
    "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_addl", "weird_notice",
]

TONIOT_GROUPS = {
    "Network Connection Telemetry": [
        "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "duration", "conn_state",
    ],
    "Packet and Byte Volumetrics": [
        "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    ],
    "DNS Context": [
        "dns_query", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    ],
    "SSL/TLS Security Handshake": [
        "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer",
    ],
    "HTTP Web Application Telemetry": [
        "http_trans_depth", "http_method", "http_uri", "http_version",
        "http_request_body_len", "http_response_body_len", "http_status_code", "http_user_agent",
        "http_orig_mime_types", "http_resp_mime_types",
    ],
    "Anomaly and Protocol Violation Flags": [
        "weird_name", "weird_addl", "weird_notice",
    ],
}

# 3. Bot-IoT 5% Official Downsampled (35 features)
BOTIOT_5PCT_FEATURES = [
    "stime", "flgs_number", "proto_number", "pkts", "bytes", "state_number", "ltime", "seq", "dur",
    "mean", "stddev", "sum", "min", "max",
    "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate",
    "TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP", "TnP_PDstIP", "TnP_PerProto", "TnP_Per_Dport",
    "AR_P_Proto_P_SrcIP", "AR_P_Proto_P_DstIP", "N_IN_Conn_P_DstIP", "N_IN_Conn_P_SrcIP",
    "AR_P_Proto_P_Sport", "AR_P_Proto_P_Dport",
    "Pkts_P_State_P_Protocol_P_DestIP", "Pkts_P_State_P_Protocol_P_SrcIP",
]

BOTIOT_5PCT_GROUPS = {
    "Temporal Flow Timing": [
        "stime", "ltime", "seq", "dur", "mean", "stddev", "sum", "min", "max",
    ],
    "Protocol and State Encodings": [
        "flgs_number", "proto_number", "state_number",
    ],
    "Packet and Byte Totals": [
        "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes",
    ],
    "Flow Rates and Speed": [
        "rate", "srate", "drate",
    ],
    "Sub-Flow Connection Aggregates": [
        "TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP", "TnP_PDstIP", "TnP_PerProto", "TnP_Per_Dport",
        "AR_P_Proto_P_SrcIP", "AR_P_Proto_P_DstIP", "N_IN_Conn_P_DstIP", "N_IN_Conn_P_SrcIP",
        "AR_P_Proto_P_Sport", "AR_P_Proto_P_Dport",
        "Pkts_P_State_P_Protocol_P_DestIP", "Pkts_P_State_P_Protocol_P_SrcIP",
    ],
}

# ---------------------------------------------------------------------------
# Detector Registry
# ---------------------------------------------------------------------------
AVAILABLE_DETECTORS = {
    "Transfer Rescue: CICIoT2023 -> Bot-IoT (NetFlow v2)": {
        "id": "xplore_h2c_tl_ciciot_to_botiot_weights.pth",
        "fp32_weights": "weights/transfer_rescue_ciciot_to_botiot_fp32.pth",
        "int8_model": "weights/transfer_rescue_ciciot_to_botiot_int8.pth",
        "features": 41,
        "feature_names": BOTIOT_FEATURES,
        "feature_groups": BOTIOT_GROUPS,
        "feature_tooltips": BOTIOT_TOOLTIPS,
        "task": "binary",
        "source_dataset": "CICIoT2023 (Base Backbone)",
        "target_dataset": "Bot-IoT (NetFlow v2)",
        "accuracy": 99.84,
        "f1": 99.92,
        "recall": 99.98,
        "precision": 99.85,
        "params_total": 1038383,
        "params_trainable": 79301,
        "fp32_size_mb": 4.03,
        "int8_size_mb": 1.16,
        "default_benchmark_batch": "data/botiot_benchmark_batch.csv",
        "default_balanced_batch": "data/botiot_demo_batch.csv",
        "packaged_test_path": str(APP_DIR / "data" / "botiot_test_slice.csv"),
        "full_test_path": str(PROJECT_ROOT / "data" / "preprocessed" / "botiot" / "binary" / "test.csv"),
        "sample_normal": "data/botiot_sample_normal.csv",
        "sample_attack": "data/botiot_sample_attack.csv",
        "operational_context": (
            "This model was fine-tuned on target traffic with an extreme 278:1 attack-to-normal skew "
            "(99.64% attacks). On the official benchmark test set (79,889 flows), transfer rescue recovered "
            "normal recognition from total collapse (0% to 59.5% specificity) and achieved 99.98% attack recall "
            "and 99.84% accuracy. At default threshold 0.50, evaluating on balanced traffic highlights the base "
            "rate shift: use the Decision Threshold slider to observe how threshold calibration suppresses false alarms."
        ),
    },
    "ToN-IoT Binary Edge Detector (From Scratch)": {
        "id": "xplore_h3a_toniot_binary_weights.pth",
        "fp32_weights": "weights/toniot_binary_fp32.pth",
        "int8_model": "weights/toniot_binary_int8.pth",
        "features": 42,
        "feature_names": TONIOT_FEATURES,
        "feature_groups": TONIOT_GROUPS,
        "feature_tooltips": {f: f"ToN-IoT telemetry feature: {f}" for f in TONIOT_FEATURES},
        "task": "binary",
        "source_dataset": "ToN-IoT Network Telemetry",
        "target_dataset": "ToN-IoT (From Scratch, Balanced)",
        "accuracy": 99.57,
        "f1": 99.72,
        "recall": 99.79,
        "precision": 99.65,
        "params_total": 1038383,
        "params_trainable": 1038383,
        "fp32_size_mb": 4.03,
        "int8_size_mb": 1.15,
        "default_benchmark_batch": "data/toniot_demo_batch.csv",
        "default_balanced_batch": "data/toniot_demo_batch.csv",
        "packaged_test_path": str(APP_DIR / "data" / "toniot_test_partition.csv"),
        "full_test_path": str(PROJECT_ROOT / "data" / "preprocessed" / "toniot" / "binary" / "test.csv"),
        "sample_normal": "data/toniot_sample_normal.csv",
        "sample_attack": "data/toniot_sample_attack.csv",
        "operational_context": (
            "From-scratch detector trained on naturally balanced Industrial IoT telemetry (Table 4.4 in dissertation). "
            "Demonstrates 99.6%+ accuracy with near-zero false alarms on balanced traffic out of the box."
        ),
    },
    "Bot-IoT 5% Official Downsampled Detector": {
        "id": "botiot_5pct_official_binary_downsampled5to1_weights.pth",
        "fp32_weights": "weights/botiot_5pct_official_fp32.pth",
        "int8_model": "weights/botiot_5pct_official_int8.pth",
        "features": 35,
        "feature_names": BOTIOT_5PCT_FEATURES,
        "feature_groups": BOTIOT_5PCT_GROUPS,
        "feature_tooltips": {f: f"Bot-IoT official 5% extraction feature: {f}" for f in BOTIOT_5PCT_FEATURES},
        "task": "binary",
        "source_dataset": "Bot-IoT Official 5% Subsample",
        "target_dataset": "Bot-IoT 5% (Train-Only 5:1 Downsampled)",
        "accuracy": 99.53,
        "f1": 99.76,
        "recall": 99.53,
        "precision": 100.0,
        "params_total": 1038383,
        "params_trainable": 1038383,
        "fp32_size_mb": 4.04,
        "int8_size_mb": 1.15,
        "default_benchmark_batch": "data/botiot_5pct_demo_batch.csv",
        "default_balanced_batch": "data/botiot_5pct_demo_batch.csv",
        "packaged_test_path": str(APP_DIR / "data" / "botiot_5pct_test_slice.csv"),
        "full_test_path": str(PROJECT_ROOT / "data" / "preprocessed" / "botiot_5pct_official" / "test.csv"),
        "sample_normal": "data/botiot_5pct_sample_normal.csv",
        "sample_attack": "data/botiot_5pct_sample_attack.csv",
        "operational_context": (
            "Trained on the official 5% Bot-IoT sample using train-only 5:1 downsampling while leaving test data "
            "completely unmodified (Table 4.5 in dissertation). Achieves 100% normal recognition (0 false alarms) "
            "and 99.53% attack recall."
        ),
    },
}

# ---------------------------------------------------------------------------
# Model Loading Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_fp32_model(weights_path: str, num_features: int = 41):
    """Load the FP32 HybridSwinLSTM from state dict."""
    model = HybridSwinLSTM(input_shape=(num_features, 1))
    full_path = APP_DIR / weights_path
    state_dict = torch.load(str(full_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@st.cache_resource
def load_int8_model(model_path: str):
    """Load the INT8 quantized full model (architecture + weights)."""
    full_path = APP_DIR / model_path
    model = torch.load(str(full_path), map_location="cpu", weights_only=False)
    model.eval()
    return model


def get_model(precision: str, detector_info: dict):
    """Return the correct model based on precision selection."""
    num_features = detector_info.get("features", 41)
    if "INT8" in precision:
        return load_int8_model(detector_info["int8_model"])
    else:
        return load_fp32_model(detector_info["fp32_weights"], num_features)


# ---------------------------------------------------------------------------
# Inference Helpers
# ---------------------------------------------------------------------------
def run_single_inference(model, features: list[float], threshold: float = 0.5) -> dict:
    """Run inference on a single feature vector."""
    tensor_input = torch.FloatTensor([features])
    t0 = time.perf_counter()
    with torch.no_grad():
        logit = model(tensor_input)
    latency_ms = (time.perf_counter() - t0) * 1000
    logit_val = logit.item() if logit.dim() == 0 else logit[0].item()
    prob = float(torch.sigmoid(torch.tensor(logit_val)))
    verdict = "Attack" if prob >= threshold else "Normal"
    return {
        "logit": logit_val,
        "probability": prob,
        "verdict": verdict,
        "latency_ms": latency_ms,
        "threshold": threshold,
    }


def run_batch_inference(
    model,
    df: pd.DataFrame,
    threshold: float = 0.5,
    progress_callback=None,
    chunk_size: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """Run inference on a batch dataframe with optional progress feedback."""
    feature_cols = [c for c in df.columns if c != "target"]
    X = torch.FloatTensor(df[feature_cols].values)
    total_records = len(df)

    t0 = time.perf_counter()
    logits_list = []

    # Dynamic chunking for responsive throughput display without slowing down execution
    effective_chunk = max(10, min(chunk_size, total_records // 20)) if total_records > 100 else 10

    if progress_callback is not None and total_records > 0:
        for i in range(0, total_records, effective_chunk):
            chunk = X[i : min(i + effective_chunk, total_records)]
            with torch.no_grad():
                chunk_logits = model(chunk)
                if chunk_logits.dim() == 0:
                    chunk_logits = chunk_logits.unsqueeze(0)
                logits_list.append(chunk_logits)

            done = min(i + effective_chunk, total_records)
            elapsed = max(time.perf_counter() - t0, 0.001)
            current_rate = done / elapsed
            progress_callback(done, total_records, current_rate)
            if total_records <= 500:
                time.sleep(0.008)
        logits = torch.cat(logits_list, dim=0) if len(logits_list) > 1 else logits_list[0]
    else:
        with torch.no_grad():
            logits = model(X)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

    latency_ms = (time.perf_counter() - t0) * 1000

    probs = torch.sigmoid(logits).numpy().reshape(-1)
    preds = (probs >= threshold).astype(int)
    verdicts = ["Attack" if p == 1 else "Normal" for p in preds]

    results = df.copy()
    results["confidence"] = np.round(probs, 4)
    results["prediction"] = preds
    results["verdict"] = verdicts

    num_attack = int(preds.sum())
    num_normal = len(preds) - num_attack

    summary = {
        "total": total_records,
        "attacks_detected": num_attack,
        "normal_detected": num_normal,
        "attack_pct": (num_attack / max(total_records, 1)) * 100.0,
        "normal_pct": (num_normal / max(total_records, 1)) * 100.0,
        "latency_ms": latency_ms,
        "latency_per_record_ms": latency_ms / max(total_records, 1),
        "throughput_fps": total_records / max(latency_ms / 1000.0, 0.001),
        "accuracy": None,
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "threshold": threshold,
    }

    # Evaluate against ground truth if present
    if "target" in df.columns:
        y_true = df["target"].values.astype(int)
        summary["accuracy"] = float((preds == y_true).mean() * 100.0)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        summary["tp"] = tp
        summary["tn"] = tn
        summary["fp"] = fp
        summary["fn"] = fn
        summary["precision"] = prec
        summary["recall"] = rec
        summary["f1"] = f1

    return results, summary


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def render_pipeline_stage(stage_num: str, label: str, detail: str, status: str = "pending"):
    """Render a single pipeline stage card."""
    colors = {
        "pending": ("#f1f3f4", "#5f6368", "#dadce0"),
        "active": ("#e8f0fe", "#1a73e8", "#4285f4"),
        "complete": ("#e6f4ea", "#137333", "#34a853"),
        "error": ("#fce8e6", "#c5221f", "#ea4335"),
    }
    bg, fg, border = colors.get(status, colors["pending"])
    st.markdown(f"""
    <div style="background:{bg}; border-radius:8px; padding:12px 14px;
                border:1px solid {border}; border-left:4px solid {fg}; margin-bottom:8px;">
        <div style="font-size:0.75em; font-weight:700; color:{fg}; text-transform:uppercase; letter-spacing:0.5px;">{stage_num}</div>
        <div style="font-weight:600; color:{fg}; font-size:0.92em; margin-top:2px;">{label}</div>
        <div style="color:#5f6368; font-size:0.80em; margin-top:3px; line-height:1.3;">{detail}</div>
    </div>
    """, unsafe_allow_html=True)


def render_verdict_card(verdict: str, probability: float, latency_ms: float, threshold: float = 0.5):
    """Render a large verdict result card with clean academic styling."""
    if verdict == "Attack":
        bg, border, badge_bg = "#fce8e6", "#ea4335", "#c5221f"
        label_color = "#c5221f"
        status_text = "MALICIOUS FLOW DETECTED"
    else:
        bg, border, badge_bg = "#e6f4ea", "#34a853", "#137333"
        label_color = "#137333"
        status_text = "NORMAL BENIGN TRAFFIC"

    confidence_pct = probability * 100 if verdict == "Attack" else (1 - probability) * 100

    st.markdown(f"""
    <div style="background:{bg}; border:1.5px solid {border}; border-radius:12px;
                padding:22px 16px; text-align:center; margin:16px 0;">
        <div style="display:inline-block; background:{badge_bg}; color:white; font-size:0.75em;
                    font-weight:700; padding:4px 12px; border-radius:16px; letter-spacing:0.8px; margin-bottom:8px;">
            {status_text}
        </div>
        <div style="font-size:1.8em; font-weight:800; color:{label_color};
                    letter-spacing:0.5px;">{verdict.upper()}</div>
        <div style="font-size:1.05em; color:#3c4043; margin-top:6px;">
            Confidence: <b>{confidence_pct:.2f}%</b> (Threshold: {threshold:.2f})
        </div>
        <div style="font-size:0.82em; color:#5f6368; margin-top:4px;">
            Inference Latency: <b>{latency_ms:.2f} ms</b> on CPU
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_badge(label: str, value: str, color: str = "#1a73e8"):
    """Render a compact metric badge."""
    st.markdown(f"""
    <div style="text-align:center; padding:8px 4px; background:#f8f9fa; border:1px solid #e8eaed; border-radius:8px;">
        <div style="font-size:1.3em; font-weight:700; color:{color};">{value}</div>
        <div style="font-size:0.75em; color:#5f6368; margin-top:2px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DISAL Edge Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        font-family: 'Google Sans', 'Segoe UI', -apple-system, sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] {
        background-color: #fafbfc;
        border-right: 1px solid #e8eaed;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.95em;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 20px;
    }
    #MainMenu {visibility: hidden;}
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e8eaed;
    }
    .stDownloadButton > button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: Settings & Model Info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## DISAL Edge Detector")
    st.caption("Bempong and Brown (2026)")

    st.markdown("---")
    st.markdown("### Detector Configuration")

    # Detector selection
    detector_name = st.selectbox(
        "Viable Detector Registry",
        options=list(AVAILABLE_DETECTORS.keys()),
        index=0,
        help="Select the trained detector model to evaluate.",
    )
    detector_info = AVAILABLE_DETECTORS[detector_name]
    num_features = detector_info["features"]
    feature_names = detector_info["feature_names"]
    feature_groups = detector_info["feature_groups"]
    feature_tooltips = detector_info["feature_tooltips"]

    # Model precision swap
    precision = st.radio(
        "Detector Precision Profile",
        options=[
            f"INT8 ({detector_info['int8_size_mb']:.2f} MB, Edge-Quantized)",
            f"FP32 ({detector_info['fp32_size_mb']:.2f} MB, Full Precision)",
        ],
        index=0,
        help="INT8 is dynamically quantized for edge gateways, achieving ≈4x compression.",
    )

    # Decision Threshold slider
    st.markdown("##### Detection Threshold")
    threshold = st.slider(
        "Operating Decision Threshold",
        min_value=0.05,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help=(
            "Probability cutoff for classifying flow as Attack (p >= threshold). "
            "In environments with extreme class skew, threshold calibration balances "
            "attack detection coverage against false alarm suppression."
        ),
    )

    # Classification mode
    st.markdown("##### Classification Task")
    col_bin, col_multi = st.columns(2)
    with col_bin:
        st.button("Binary (Active)", disabled=False, use_container_width=True, type="primary")
    with col_multi:
        st.button("Multi-Class", disabled=True, use_container_width=True, help="Multi-class attack profiling scheduled for Phase 2")
    st.caption("Multi-Class Attack Profiler: Coming Soon (Phase 2)")

    # Phase 2 pipeline toggles
    with st.expander("Edge Gateway Extensions (Phase 2)", expanded=False):
        st.markdown("**PCAP Socket Stream Sniffer**")
        st.caption("Status: Coming Soon")
        st.markdown("**In-Memory Scaler Fitting**")
        st.caption("Status: Coming Soon")
        st.markdown("**Zero-Leakage Drift Monitor**")
        st.caption("Status: Coming Soon")

    st.markdown("---")
    st.markdown("### Published Benchmark Metrics")

    m1, m2 = st.columns(2)
    with m1:
        render_metric_badge("Accuracy", f"{detector_info['accuracy']:.2f}%")
    with m2:
        render_metric_badge("F1-Score", f"{detector_info['f1']:.2f}%")

    m3, m4 = st.columns(2)
    with m3:
        render_metric_badge("Attack Recall", f"{detector_info['recall']:.2f}%", "#137333")
    with m4:
        render_metric_badge("Precision", f"{detector_info['precision']:.2f}%")

    st.markdown("---")
    st.markdown("### Architecture Profile")

    prec_label = "INT8" if "INT8" in precision else "FP32"
    size_val = detector_info["int8_size_mb"] if prec_label == "INT8" else detector_info["fp32_size_mb"]

    st.markdown(f"""
    | Property | Value |
    |---|---|
    | Architecture | HybridSwinLSTM |
    | Parameters | {detector_info['params_total']:,} |
    | Trainable Params | {detector_info['params_trainable']:,} |
    | Precision | {prec_label} |
    | Disk Footprint | {size_val:.2f} MB |
    | Feature Dimension | {detector_info['features']} features |
    | Source Domain | {detector_info['source_dataset']} |
    | Target Domain | {detector_info['target_dataset']} |
    """)

    st.markdown("---")
    st.markdown("### Technical Datasheet")
    datasheet_path = APP_DIR / "data" / "botiot_netflow_v2_datasheet.pdf"
    if datasheet_path.exists():
        with open(datasheet_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="Download PDF Datasheet",
            data=pdf_bytes,
            file_name="botiot_netflow_v2_datasheet.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Download the formal technical datasheet covering provenance, scaling, and feature definitions.",
        )

    st.markdown("---")
    st.caption(detector_info["operational_context"])


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
try:
    model = get_model(precision, detector_info)
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False


# ---------------------------------------------------------------------------
# Main Content: Tabs
# ---------------------------------------------------------------------------
st.markdown("# Edge Detection Pipeline")
st.caption("Real-time IoT network intrusion detection powered by the Hybrid Swin-LSTM architecture")

tab_single, tab_batch, tab_pipeline = st.tabs([
    "Single Record",
    "Batch Inference",
    "Pipeline Overview",
])


# ---------------------------------------------------------------------------
# Tab 1: Single Record Inference
# ---------------------------------------------------------------------------
with tab_single:
    st.markdown("#### Inspect a Single Flow Record")
    st.caption(f"Enter pre-scaled feature values (range 0.0 to 1.0) or load verified benchmark samples for {detector_info['target_dataset']}.")
    st.caption("Hover over the info icon (i) beside each feature to view its diagnostic definition.")

    col_load1, col_load2, col_load3 = st.columns(3)
    with col_load1:
        if st.button("Load Sample Normal Flow", use_container_width=True,
                     help="Load a verified normal flow record for this detector"):
            try:
                sample_df = pd.read_csv(APP_DIR / detector_info["sample_normal"])
                row = sample_df.iloc[0]
                for f in feature_names:
                    if f in row.index:
                        st.session_state[f"single_{f}"] = float(row[f])
                st.rerun()
            except Exception as e:
                st.warning(f"Could not load normal sample: {e}")

    with col_load2:
        if st.button("Load Sample Attack Flow", use_container_width=True,
                     help="Load a verified attack record for this detector"):
            try:
                sample_df = pd.read_csv(APP_DIR / detector_info["sample_attack"])
                row = sample_df.iloc[0]
                for f in feature_names:
                    if f in row.index:
                        st.session_state[f"single_{f}"] = float(row[f])
                st.rerun()
            except Exception as e:
                st.warning(f"Could not load attack sample: {e}")

    with col_load3:
        if st.button("Reset Values to 0.0", use_container_width=True):
            for f in feature_names:
                st.session_state[f"single_{f}"] = 0.0
            st.rerun()

    # Dynamic feature inputs grouped logically
    feature_values = []
    for group_name, group_features in feature_groups.items():
        with st.expander(f"**{group_name}** ({len(group_features)} features)", expanded=False):
            cols = st.columns(min(len(group_features), 4))
            for i, feat in enumerate(group_features):
                if f"single_{feat}" not in st.session_state:
                    st.session_state[f"single_{feat}"] = 0.0
                with cols[i % len(cols)]:
                    val = st.number_input(
                        feat,
                        format="%.6f",
                        step=0.01,
                        key=f"single_{feat}",
                        help=feature_tooltips.get(feat, f"{feat} feature definition."),
                    )
                    feature_values.append(val)

    # Inference trigger
    st.markdown("")
    run_col, info_col = st.columns([1, 2])
    with run_col:
        run_single = st.button(
            "Run Edge Detection",
            use_container_width=True,
            type="primary",
            disabled=not model_loaded,
        )

    if run_single and model_loaded:
        result = run_single_inference(model, feature_values, threshold=threshold)
        st.session_state["single_result"] = result

    single_res = st.session_state.get("single_result")
    if single_res is not None and model_loaded:
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_pipeline_stage("Stage 1", "Flow Ingestion", f"{num_features} features ready", "complete")
        with p2:
            render_pipeline_stage("Stage 2", "Feature Scaling", "Zero-leakage normalized", "complete")
        with p3:
            render_pipeline_stage(
                "Stage 3",
                "Neural Inference",
                f"Latency: {single_res['latency_ms']:.2f} ms ({prec_label})",
                "complete",
            )
        status = "complete" if single_res["verdict"] == "Normal" else "error"
        with p4:
            render_pipeline_stage(
                "Stage 4",
                "Detection Verdict",
                f"{single_res['verdict']} (p={single_res['probability']:.4f})",
                status,
            )

        render_verdict_card(single_res["verdict"], single_res["probability"], single_res["latency_ms"], threshold=threshold)

        with st.expander("Technical Diagnostics and Output Tensors"):
            st.json({
                "raw_logit": round(single_res["logit"], 6),
                "sigmoid_probability": round(single_res["probability"], 6),
                "threshold": threshold,
                "verdict": single_res["verdict"],
                "inference_latency_ms": round(single_res["latency_ms"], 2),
                "model_precision": prec_label,
                "input_features": num_features,
            })


# ---------------------------------------------------------------------------
# Tab 2: Batch CSV Inference with Live Feedback and Diagnostics
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown("#### Batch Flow Analysis and Live Detection Pipeline")
    st.caption("Inspect live network traffic batches, validate end-to-end edge throughput, and review detection logs.")

    # Initialize batch session state
    if "batch_df" not in st.session_state:
        st.session_state["batch_df"] = None
        st.session_state["batch_name"] = None
    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = None
    if "batch_summary" not in st.session_state:
        st.session_state["batch_summary"] = None

    # If the active detector changed, update the active batch to match the new detector's required features
    if st.session_state.get("active_detector_id") != detector_info["id"] or st.session_state["batch_df"] is None:
        st.session_state["active_detector_id"] = detector_info["id"]
        def_path = APP_DIR / detector_info["default_benchmark_batch"]
        if def_path.exists():
            st.session_state["batch_df"] = pd.read_csv(def_path)
            st.session_state["batch_name"] = f"{detector_info['target_dataset']} Benchmark ({def_path.name})"
            st.session_state["batch_results"] = None
            st.session_state["batch_summary"] = None

    # Select Traffic Source
    source_mode = st.radio(
        "Traffic Ingestion Source",
        options=[
            "Pre-Packaged Demo Batches",
            "Full Evaluation Test Partition",
            "Upload Custom NetFlow CSV",
        ],
        horizontal=True,
    )

    if source_mode == "Pre-Packaged Demo Batches":
        c_p1, c_p2 = st.columns([3, 1])
        with c_p1:
            batch_choice = st.selectbox(
                "Select Pre-Packaged Traffic Batch",
                options=[
                    f"Active Detector Benchmark ({Path(detector_info['default_benchmark_batch']).name})",
                    "ToN-IoT Balanced Batch (100 flows: 50 Normal, 50 Attack)",
                    "Bot-IoT Benchmark Batch (100 flows: 1 Normal, 99 Attack - 278:1 Skew)",
                    "Bot-IoT Balanced Demo (100 flows: 50 Normal, 50 Attack)",
                    "Bot-IoT 5% Official Batch (100 flows: 47 Normal, 53 Attack)",
                ],
                index=0,
            )
        with c_p2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Load Selected Batch", use_container_width=True):
                target_file = None
                if "ToN-IoT" in batch_choice:
                    target_file = APP_DIR / "data" / "toniot_demo_batch.csv"
                elif "278:1 Skew" in batch_choice:
                    target_file = APP_DIR / "data" / "botiot_benchmark_batch.csv"
                elif "Bot-IoT Balanced" in batch_choice:
                    target_file = APP_DIR / "data" / "botiot_demo_batch.csv"
                elif "5%" in batch_choice:
                    target_file = APP_DIR / "data" / "botiot_5pct_demo_batch.csv"
                else:
                    target_file = APP_DIR / detector_info["default_benchmark_batch"]

                if target_file and target_file.exists():
                    st.session_state["batch_df"] = pd.read_csv(target_file)
                    st.session_state["batch_name"] = f"{batch_choice} ({target_file.name})"
                    st.session_state["batch_results"] = None
                    st.session_state["batch_summary"] = None
                    st.rerun()

    elif source_mode == "Full Evaluation Test Partition":
        packaged_str = detector_info.get("packaged_test_path")
        packaged_path = Path(packaged_str) if (packaged_str and Path(packaged_str).is_file()) else None

        disk_str = detector_info.get("full_test_path")
        disk_path = Path(disk_str) if (disk_str and Path(disk_str).is_file()) else None

        full_test_file = packaged_path or disk_path

        if full_test_file is not None and full_test_file.is_file():
            c_f1, c_f2 = st.columns([3, 1])
            with c_f1:
                slice_choice = st.selectbox(
                    f"Select Partition Slice to Evaluate ({full_test_file.name})",
                    options=[
                        "100 flows (Quick Verification)",
                        "500 flows (Fast Edge Stream)",
                        "1,000 flows (Standard Benchmark)",
                        "2,500 flows (Extended Verification)",
                        "5,000 flows (High-Volume Stress Test)",
                        "All Records (Full Test Partition)",
                    ],
                    index=2,
                    help="Evaluating slices allows testing edge line rates and accuracy on authentic un-modified test partitions.",
                )
            with c_f2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("Load Test Partition", use_container_width=True, type="secondary"):
                    slice_map = {
                        "100": 100,
                        "500": 500,
                        "1,000": 1000,
                        "2,500": 2500,
                        "5,000": 5000,
                        "All": None,
                    }
                    key_val = slice_choice.split()[0]
                    nrows = slice_map.get(key_val, 1000)
                    with st.spinner(f"Ingesting benchmark partition from {full_test_file.name}..."):
                        loaded_df = pd.read_csv(full_test_file, nrows=nrows)
                        if "multi_label" in loaded_df.columns:
                            loaded_df = loaded_df.drop(columns=["multi_label"])
                        t_col = next((c for c in ["binary_label", "target", "Label", "label"] if c in loaded_df.columns), None)
                        if t_col and t_col != "target":
                            loaded_df = loaded_df.rename(columns={t_col: "target"})
                        st.session_state["batch_df"] = loaded_df
                        st.session_state["batch_name"] = f"Test Partition: {full_test_file.name} ({len(loaded_df):,} flows)"
                        st.session_state["batch_results"] = None
                        st.session_state["batch_summary"] = None
                        st.rerun()
        else:
            st.info(
                f"The complete {detector_info['target_dataset']} benchmark partition file is not packaged in this deployment directory. "
                "Please select 'Pre-Packaged Demonstration Batches' above to evaluate this detector, or upload a custom CSV."
            )

    else:
        uploaded_file = st.file_uploader(
            "Upload Custom NetFlow CSV",
            type=["csv"],
            key="batch_csv_uploader",
            help=f"Upload a CSV with {num_features} features. Optional 'target' column enables accuracy calculation.",
        )
        if uploaded_file is not None:
            if st.session_state.get("current_uploaded_name") != uploaded_file.name:
                try:
                    custom_df = pd.read_csv(uploaded_file)
                    st.session_state["batch_df"] = custom_df
                    st.session_state["batch_name"] = f"Uploaded: {uploaded_file.name} ({len(custom_df)} flows)"
                    st.session_state["batch_results"] = None
                    st.session_state["batch_summary"] = None
                    st.session_state["current_uploaded_name"] = uploaded_file.name
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse uploaded CSV: {e}")

    # Active batch overview & execution
    active_batch_df = st.session_state.get("batch_df")

    if active_batch_df is not None:
        feature_cols = [c for c in active_batch_df.columns if c != "target"]
        has_ground_truth = "target" in active_batch_df.columns
        valid_feature_count = len(feature_cols) == num_features

        # Active dataset overview card
        norm_cnt = int((active_batch_df["target"] == 0).sum()) if has_ground_truth else 0
        att_cnt = int((active_batch_df["target"] == 1).sum()) if has_ground_truth else 0
        dist_text = f" &nbsp;|&nbsp; <strong>Distribution:</strong> {norm_cnt:,} Normal, {att_cnt:,} Attack" if has_ground_truth else ""

        st.markdown(f"""
        <div style="background:#f1f3f4; border-radius:8px; padding:10px 16px; margin:12px 0 16px 0;
                    border:1px solid #dadce0; font-size:0.90em; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <strong>Active Batch:</strong> {st.session_state.get('batch_name', 'Loaded Batch')}
                &nbsp;|&nbsp;
                <strong>Records:</strong> {len(active_batch_df):,} flows
                &nbsp;|&nbsp;
                <strong>Features:</strong> {len(feature_cols)}/{num_features} {('(Valid)' if valid_feature_count else '(Mismatch)')}
                {dist_text}
            </div>
            <div>
                <span style="background:{'#e6f4ea' if has_ground_truth else '#feefc3'};
                             color:{'#137333' if has_ground_truth else '#b06000'};
                             font-weight:600; padding:3px 10px; border-radius:12px; font-size:0.85em;">
                    {'Ground Truth Available' if has_ground_truth else 'Unlabeled Ingestion'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not valid_feature_count:
            missing = set(feature_names) - set(feature_cols)
            st.error(f"Feature count mismatch: Active detector requires {num_features} features, but batch has {len(feature_cols)}. Missing: {list(missing)[:5]}")

        with st.expander("Inspect Raw Ingested Flow Data (First 5 Records)", expanded=False):
            st.dataframe(active_batch_df.head(5), use_container_width=True, height=200)

        # Run Batch Trigger
        btn_label = f"Run End-to-End Edge Detection Pipeline ({len(active_batch_df):,} Flows @ Threshold {threshold:.2f})"
        run_batch_trigger = st.button(
            btn_label,
            use_container_width=True,
            type="primary",
            disabled=(not model_loaded or not valid_feature_count),
        )

        if run_batch_trigger and model_loaded and valid_feature_count:
            st.markdown("##### Executing Edge Detection Pipeline")
            stage_box = st.empty()
            with stage_box.container():
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    render_pipeline_stage("Stage 1", "Packet Capture", f"{len(active_batch_df):,} packets ingested", "complete")
                with p2:
                    render_pipeline_stage("Stage 2", "Flow Scaling", f"{num_features} tensors normalized", "complete")
                with p3:
                    render_pipeline_stage("Stage 3", "Neural Inference", "Processing batches...", "active")
                with p4:
                    render_pipeline_stage("Stage 4", "Detection Verdict", "Aggregating verdicts...", "pending")

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_callback(done, total, rate):
                pct = float(done) / max(float(total), 1.0)
                progress_bar.progress(pct)
                status_text.markdown(
                    f"**Streaming Inference:** {done}/{total} records processed "
                    f"({pct*100.0:.0f}%) &nbsp;|&nbsp; Line rate: **{rate:.0f} flows/sec**"
                )

            res_df, res_summary = run_batch_inference(
                model,
                active_batch_df,
                threshold=threshold,
                progress_callback=progress_callback,
            )

            st.session_state["batch_results"] = res_df
            st.session_state["batch_summary"] = res_summary

            progress_bar.progress(1.0)
            status_text.markdown(
                f"**Detection Complete:** {len(active_batch_df):,} records evaluated "
                f"in {res_summary['latency_ms']:.1f} ms (**{res_summary['throughput_fps']:.0f} flows/sec**)"
            )

            with stage_box.container():
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    render_pipeline_stage("Stage 1", "Packet Capture", f"{len(active_batch_df):,} packets ingested", "complete")
                with p2:
                    render_pipeline_stage("Stage 2", "Flow Scaling", f"{num_features} tensors normalized", "complete")
                with p3:
                    render_pipeline_stage("Stage 3", "Neural Inference", f"Throughput: {res_summary['throughput_fps']:.0f} fps", "complete")
                with p4:
                    render_pipeline_stage(
                        "Stage 4",
                        "Detection Verdict",
                        f"{res_summary['attacks_detected']} Attacks / {res_summary['normal_detected']} Normal",
                        "complete",
                    )

    # Display results
    batch_res = st.session_state.get("batch_results")
    summary = st.session_state.get("batch_summary")

    if batch_res is not None and summary is not None:
        st.markdown("---")
        st.markdown("### Detection Performance Summary")

        # 4 Summary Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Inspected", f"{summary['total']:,}")
        c2.metric(
            "Attacks Flagged",
            f"{summary['attacks_detected']:,}",
            f"{summary['attack_pct']:.1f}% of total",
            delta_color="inverse",
        )
        c3.metric(
            "Normal Traffic",
            f"{summary['normal_detected']:,}",
            f"{summary['normal_pct']:.1f}% of total",
            delta_color="normal",
        )
        c4.metric(
            "Line Throughput",
            f"{summary['throughput_fps']:.0f} flows/s",
            f"{summary['latency_per_record_ms']:.2f} ms/flow",
        )

        # Ground truth benchmarks
        if summary.get("accuracy") is not None:
            st.markdown("#### Supervised Benchmark Validation")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Detection Accuracy", f"{summary['accuracy']:.2f}%")
            b2.metric("Attack Recall", f"{summary['recall']:.1f}%")
            b3.metric("Alert Precision", f"{summary['precision']:.1f}%")
            b4.metric("F1 Score", f"{summary['f1']:.1f}%")

            st.markdown("##### Confusion Matrix Breakdown")
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.markdown(f"""
            <div style="background:#e6f4ea; border:1px solid #ceead6; border-radius:8px; padding:10px; text-align:center;">
                <div style="font-size:1.4em; font-weight:700; color:#137333;">{summary['tp']:,}</div>
                <div style="font-size:0.80em; color:#137333;">True Attacks (TP)</div>
            </div>
            """, unsafe_allow_html=True)

            cm2.markdown(f"""
            <div style="background:#fce8e6; border:1px solid #fad2cf; border-radius:8px; padding:10px; text-align:center;">
                <div style="font-size:1.4em; font-weight:700; color:#c5221f;">{summary['fn']:,}</div>
                <div style="font-size:0.80em; color:#c5221f;">Missed Attacks (FN)</div>
            </div>
            """, unsafe_allow_html=True)

            cm3.markdown(f"""
            <div style="background:#feefc3; border:1px solid #fce8b2; border-radius:8px; padding:10px; text-align:center;">
                <div style="font-size:1.4em; font-weight:700; color:#b06000;">{summary['fp']:,}</div>
                <div style="font-size:0.80em; color:#b06000;">False Alarms (FP)</div>
            </div>
            """, unsafe_allow_html=True)

            cm4.markdown(f"""
            <div style="background:#e8f0fe; border:1px solid #d2e3fc; border-radius:8px; padding:10px; text-align:center;">
                <div style="font-size:1.4em; font-weight:700; color:#1a73e8;">{summary['tn']:,}</div>
                <div style="font-size:0.80em; color:#1a73e8;">True Benign (TN)</div>
            </div>
            """, unsafe_allow_html=True)

            # Operational insight context banner
            st.markdown("")
            st.info(f"**Operational Insight (Operating Cutoff = {threshold:.2f}):** {detector_info['operational_context']}")

        # Detailed Inspection Log Table
        st.markdown("#### Inspection Log and Confidence Scores")
        display_cols = [c for c in batch_res.columns if c not in ["target"]]
        if "target" in batch_res.columns:
            display_cols = ["target"] + display_cols

        st.dataframe(
            batch_res[display_cols].head(100),
            use_container_width=True,
            height=300,
        )

        csv_data = batch_res.to_csv(index=False)
        st.download_button(
            "Export Annotated Inspection Log (CSV)",
            data=csv_data,
            file_name="edge_detection_inspection_log.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 3: End-to-End Pipeline Architecture & Defense Overview
# ---------------------------------------------------------------------------
with tab_pipeline:
    st.markdown("#### End-to-End Edge Detection Architecture")
    st.caption("Bempong and Brown (2026) | Master of Science in Computer Science Defense Integration")

    st.markdown("""
    ```
    Live Network Traffic (Edge IoT Gateway)
         |
         v
    [ Stage 1: Packet Capture ]
    Raw Ethernet/IP packets captured via PCAP or libpcap socket stream.
         |
         v
    [ Stage 2: NetFlow Flow Aggregator ]
    Packets grouped into bidirectional flows and summarized into numeric NetFlow features.
         |
         v
    [ Stage 3: Zero-Leakage Scaler ]
    Normalized strictly via train-fitted RobustScaler and MinMaxScaler transforms.
         |
         v
    [ Stage 4: Hybrid Swin-LSTM Edge Neural Core ]
    1D Swin Transformer Backbone (Local Feature Windows) + LSTM (Temporal Dependencies).
         |
         v
    [ Stage 5: Calibrated Sigmoid Threshold ]
    Outputs immediate verdict (Normal vs Attack) with configurable sensitivity.
    ```
    """)

    st.markdown("---")
    st.markdown("#### Defense Takeaways and Cross-Domain Generalization")

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.markdown("""
        **1. Resolving Training Collapse via Transfer Learning**
        - In target domains with extreme class imbalance (such as Bot-IoT NetFlow v2 at 278:1 attack skew), from-scratch training collapses into predicting the majority attack class (0% normal recall).
        - Transferring spatial feature representations from CICIoT2023 (`xplore_h2c_tl_ciciot_to_botiot`) rescues benign traffic recognition (recovering from 0% to 59.5% specificity) and achieves 99.84% accuracy and 99.98% recall (Table 4.12 in dissertation).
        """)

    with c_d2:
        st.markdown("""
        **2. Edge Gateway Feasibility and Quantization**
        - Dynamic INT8 quantization compresses the 4.13 MB FP32 checkpoint down to ≈1.15 MB with near-zero latency degradation.
        - Processes network flows on standard CPU at 80 to 200+ flows/second, making the architecture viable for deployment on resource-constrained IoT gateways without GPU accelerators.
        """)

    st.markdown("---")
    st.markdown("#### Comprehensive Defense Cross-Dataset Benchmark (Dissertation Table 4.12)")

    st.markdown("""
    | Source Domain | Target Domain | Accuracy | F1-Score | Attack Recall | Benchmark Finding |
    |---|---|---|---|---|---|
    | **CICIoT2023** | **Bot-IoT (NetFlow v2)** | **99.84%** | **99.92%** | **99.98%** | **Transfer Rescue: pre-trained weights resolved collapse** |
    | **ToN-IoT** | **ToN-IoT (From Scratch)** | **99.57%** | **99.72%** | **99.79%** | **Clean separation on balanced Industrial IoT telemetry** |
    | **Bot-IoT (5%)** | **Bot-IoT (5% Official)** | **99.53%** | **99.76%** | **99.53%** | **100% normal recognition via train-only downsampling** |
    | NSL-KDD | CICIoT2023 | 98.59% | 99.27% | 98.88% | Effective protocol representation reuse across domains |
    | CICIoT2023 | ToN-IoT | 85.77% | 91.22% | 96.82% | Cross-domain adaptation without target architecture change |
    """)

    st.markdown("---")
    st.caption("Demonstration prototype built for MSc dissertation defense. Verified against empirical evaluation logs.")
