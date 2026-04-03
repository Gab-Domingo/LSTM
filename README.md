# EMG Gesture Classification for Wheelchair Control

Real-time surface EMG (sEMG) gesture recognition system for wheelchair control using a compact LSTM model. The system classifies two gestures — **REST** and **FIST** — and is designed to run on a Raspberry Pi 5.

---

## Hardware

| Component | Role |
|---|---|
| **ESP32** (×2) | Reads sEMG + IMU sensors and streams data via UDP |
| **Raspberry Pi 5** | Runs real-time inference with the deployed LSTM model |

- **Device 1 (LEFT arm)** — Port 5000  
- **Device 2 (RIGHT arm)** — Port 5001  
- Both devices are collected simultaneously during training sessions

---

## Model

### Architecture
- Compact LSTM with temporal attention
- **~55,890 parameters** — model size: 0.21 MB
- Two-layer unidirectional LSTM
- Strong regularization (dropout, weight decay)

### Input Pipeline
| Parameter | Value |
|---|---|
| EMG sampling rate | 500 Hz |
| IMU sampling rate | 100 Hz |
| Window size | 100 samples |
| Stride | 25 samples |
| Sequence length | 8 windows |

### Features (per window)
- EMG envelope (RMS)
- Bandpower — 3 frequency bands
- IMU statistics — 4 features (gyro/accel magnitudes)
- vs-REST discriminative features — 6 features

### Normalization
Two-step global `StandardScaler` normalization. No per-window normalization is applied — this intentionally preserves the amplitude difference between REST and FIST, which is a key discriminating signal.

### Confidence Threshold
A high confidence threshold of **0.78** is used. This is a safety-first design choice that prioritizes avoiding false FIST activations (which could cause unintended wheelchair movement) over sensitivity.

---

## Performance

| Metric | Score |
|---|---|
| Accuracy | 92.35% |
| F1 Score | 92.29% |
| Precision | 92.49% |
| Recall | 92.35% |
| FIST Precision | 90.83% |
| FIST Recall | 96.46% |
| REST Precision | 94.74% |
| REST Recall | 86.75% |

> Deployed: 2026-03-09 — Test user: B

---

## Key Design Decisions

**LMS filtering removed**  
LMS adaptive filtering was previously used with gyroscope data for motion artifact cancellation. For wheelchair control, this was removed because wheelchair movement is part of the gesture context — not noise.

**RMS-only EMG envelope**  
Previously the envelope averaged RMS and LMS outputs. Now only RMS is used, keeping the signal cleaner and more consistent with the deployment environment.

**IMU as motion context**  
IMU features (gyro/accel magnitudes and statistics) are retained as motion context features to help the model distinguish active wheelchair movement from rest — not for noise cancellation.

---

## Repository Structure

```
emg_dataset/
│
├── notebook/
│   ├── LSTM.ipynb              # Model training notebook
│   ├── deployment-3/           # Raspberry Pi deployment package
│   │   ├── inference.py        # Main real-time inference script
│   │   ├── model_loader.py     # Model loading utilities
│   │   ├── preprocessor.py     # Real-time feature extraction
│   │   ├── window_buffer.py    # Sliding window buffer
│   │   ├── udp_server.py       # UDP server for ESP32 communication
│   │   ├── feature_monitor.py  # Feature monitoring utilities
│   │   ├── prediction_monitor.py
│   │   └── requirements.txt
│   └── model_metrics/          # Saved model config and metrics
│
├── data_collection/            # ESP32 data collection scripts
│   ├── main.py
│   ├── udp_receiver.py
│   ├── gesture_recorder.py
│   ├── session_manager.py
│   ├── data_synchronizer.py
│   ├── filters.py
│   └── config.py
│
├── data_loader/                # Dataset loading and processing
│   ├── data_processor.py
│   ├── emg_dataset.parquet
│   ├── session_loader.py
│   └── gesture_statistics.py
│
├── etl_main.py                 # ETL pipeline entry point
├── etl_extractor.py
├── etl_transformer.py
├── etl_loader.py
└── etl_requirements.txt
```
