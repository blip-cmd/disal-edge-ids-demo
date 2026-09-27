# Hybrid Model App: Edge Detection Pipeline Demo

## Phase 1: Core Inference App
- [x] Create self-contained model.py with HybridSwinLSTM architecture
- [x] Copy FP32 weights to hybrid-model-app/weights/
- [x] Run INT8 quantization and bundle quantized model (~1.21 MB)
- [x] Extract sample data from Bot-IoT test set (well-named: botiot_sample_*)
- [x] Build Streamlit app with single-record inference (feature groups, sample loaders)
- [x] Build Streamlit app with batch CSV inference (upload + demo batch)
- [x] Add pipeline overview tab with visual stage cards
- [x] Add sidebar with detector selection, FP32/INT8 toggle, model card
- [x] Add "Coming Soon" multi-class configuration button
- [x] Configure for both local and Streamlit Cloud (requirements.txt)
- [x] Create automated tests in tests/ subdirectory
- [x] Test locally with automated test suite (test_model_load.py, test_batch_inference.py)
- [x] Verify inference runs smoothly on demo batch and sample records
- [x] Add interactive hover tooltip explanations (info icon) for all 41 NetFlow features in single record view
- [x] Author and compile formal downloadable PDF datasheet for Bot-IoT NetFlow v2 source data and features
- [x] Add PDF datasheet download buttons in UI (sidebar and pipeline overview tabs)

## Phase 2: Scalers & Raw Input Mode
- [ ] Re-fit and save Bot-IoT RobustScaler + MinMaxScaler from training CSV
- [ ] Add raw-input mode (user enters unscaled values, app applies scalers)
- [ ] Add latency benchmarking display (FP32 vs INT8 comparison chart)
- [ ] Add confusion matrix visualization for batch results

## Phase 3: Full Edge Pipeline Demo
- [ ] Add PCAP upload and CICFlowMeter integration (or simulated mock)
- [ ] Add live packet capture simulation with streaming inference
- [ ] Add multi-class attack categorization (DDoS, DoS, Recon, etc.)
- [ ] Add additional viable detectors to the selector list
- [ ] Deploy to Streamlit Community Cloud with public URL
