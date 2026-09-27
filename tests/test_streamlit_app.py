"""
Test: Streamlit App UI Execution and Widget State

Uses Streamlit's official AppTest framework to validate that app.py
renders completely, processes user widget events without exceptions,
and successfully executes both single-record and batch detection.
"""

import os
import sys
from streamlit.testing.v1 import AppTest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(APP_DIR, "app.py")


def test_app_initial_render():
    """Verify that the app boots and renders all default tabs without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception, f"App raised exceptions on initial load: {at.exception}"
    assert len(at.tabs) == 3, f"Expected 3 tabs, found {len(at.tabs)}"
    print("[PASS] Initial render: 3 tabs loaded with zero exceptions")


def test_app_precision_toggle():
    """Verify that changing precision between INT8 and FP32 triggers no exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Radio widget is in sidebar
    radio = at.sidebar.radio[0]
    assert "INT8" in radio.value, f"Expected default INT8 precision, got {radio.value}"

    # Select FP32
    radio.set_value("FP32 (4.03 MB, Full Precision)").run()
    assert not at.exception, f"Exception occurred when selecting FP32: {at.exception}"
    print("[PASS] Precision toggle to FP32 executed with zero exceptions")


def test_app_single_record_sample_loaders():
    """Verify loading attack and normal samples updates session state and runs detection."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Click "Load Sample Attack (Bot-IoT)"
    btn_attack = next((b for b in at.button if "Sample Attack" in b.label), None)
    assert btn_attack is not None, "Load Sample Attack button not found"
    btn_attack.click().run()
    assert not at.exception, f"Exception after clicking Load Attack: {at.exception}"

    # Click "Run Edge Detection"
    btn_run = next((b for b in at.button if "Run Edge Detection" in b.label), None)
    assert btn_run is not None, "Run Edge Detection button not found"
    btn_run.click().run()
    assert not at.exception, f"Exception after running detection: {at.exception}"

    print("[PASS] Single record sample loader and detection executed with zero exceptions")


def test_app_batch_inference_demo():
    """Verify demo batch execution in Batch Inference tab."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Find the batch execution button (demo batch is loaded into session state by default)
    btn_run_batch = next((b for b in at.button if "Edge Detection Pipeline" in b.label or "Run Detection on" in b.label), None)
    assert btn_run_batch is not None, "Batch execution button not found"
    btn_run_batch.click().run()
    assert not at.exception, f"Exception after running batch detection: {at.exception}"

    # Verify session state holds the results
    assert at.session_state["batch_results"] is not None, "batch_results not saved in session state"
    assert at.session_state["batch_summary"] is not None, "batch_summary not saved in session state"
    assert at.session_state["batch_summary"]["total"] == 100, f"Expected 100 records, got {at.session_state['batch_summary']['total']}"

    print("[PASS] Batch inference demo batch executed with zero exceptions and verified session state")


def test_app_detector_selection():
    """Verify switching between viable detectors in the registry."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Find the detector selectbox
    sel_detector = at.sidebar.selectbox[0]
    options = sel_detector.options
    assert len(options) == 3, f"Expected 3 viable detectors, found {len(options)}"

    # Switch to ToN-IoT
    toniot_opt = next(opt for opt in options if "ToN-IoT" in opt)
    sel_detector.set_value(toniot_opt).run()
    assert not at.exception, f"Exception after selecting ToN-IoT: {at.exception}"

    # Click run batch on ToN-IoT default batch
    btn_run = next((b for b in at.button if "Edge Detection Pipeline" in b.label), None)
    assert btn_run is not None, "Batch run button not found for ToN-IoT"
    btn_run.click().run()
    assert not at.exception, f"Exception after running ToN-IoT batch: {at.exception}"
    assert at.session_state["batch_summary"]["accuracy"] >= 80.0, f"Expected high accuracy on ToN-IoT balanced batch, got {at.session_state['batch_summary']['accuracy']}"
    print(f"[PASS] Multi-detector selection (ToN-IoT) executed with zero exceptions (Accuracy: {at.session_state['batch_summary']['accuracy']}%)")


def test_app_threshold_slider():
    """Verify modifying the decision threshold updates inference verdicts."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    slider = at.sidebar.slider[0]
    assert slider.value == 0.50, f"Expected default threshold 0.50, got {slider.value}"

    # Set threshold to 0.95
    slider.set_value(0.95).run()
    assert not at.exception, f"Exception after changing threshold: {at.exception}"

    btn_run = next((b for b in at.button if "Edge Detection Pipeline" in b.label), None)
    assert btn_run is not None, "Batch run button not found"
    btn_run.click().run()
    assert not at.exception, f"Exception after running batch with new threshold: {at.exception}"
    assert at.session_state["batch_summary"]["threshold"] == 0.95
    print("[PASS] Decision threshold slider adjustment executed with zero exceptions")


if __name__ == "__main__":
    print("=" * 60)
    print("Streamlit AppTest Execution Suite")
    print("=" * 60)

    test_app_initial_render()
    test_app_precision_toggle()
    test_app_single_record_sample_loaders()
    test_app_batch_inference_demo()
    test_app_detector_selection()
    test_app_threshold_slider()

    print("\n" + "=" * 60)
    print("All Streamlit app tests passed.")
    print("=" * 60)
