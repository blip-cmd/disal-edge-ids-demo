"""
Master Test Runner for hybrid-model-app

Executes all automated test suites covering model loading, batch inference,
data validation, and Streamlit AppTest widget simulation.
"""

import os
import sys
import subprocess

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

TEST_FILES = [
    "test_model_load.py",
    "test_batch_inference.py",
    "test_streamlit_app.py",
]


def run_all():
    print("=" * 70)
    print("HYBRID MODEL APP: COMPLETE AUTOMATED TEST SUITE")
    print("=" * 70)

    all_passed = True
    for test_file in TEST_FILES:
        test_path = os.path.join(TESTS_DIR, test_file)
        print(f"\n--- Running: {test_file} ---")
        res = subprocess.run([PYTHON_EXE, test_path], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(f"[FAIL] {test_file} failed with return code {res.returncode}")
            if res.stderr:
                print("Error output:\n", res.stderr)
            all_passed = False
        else:
            print(f"[PASS] {test_file} completed successfully.")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL APP TESTS PASSED - STREAMLIT APP IS READY FOR PHASE 1")
    else:
        print("SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all())
