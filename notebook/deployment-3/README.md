# EMG Gesture Classification - Raspberry Pi Deployment

Real-time REST vs FIST gesture classification using LSTM model on Raspberry Pi.

## Overview

This deployment package contains everything needed to run real-time EMG gesture classification on a Raspberry Pi. The system receives sensor data from an ESP32 device via TCP and makes gesture predictions in real-time.

## Files

- `inference.py` - Main inference script with terminal logging
- `model_loader.py` - Model loading utilities
- `preprocessor.py` - Real-time feature extraction
- `window_buffer.py` - Sliding window buffer management
- `tcp_server.py` - TCP server for ESP32 communication
- `emg_lstm_model.pt` - Trained PyTorch model
- `deployment_config.json` - Model configuration
- `standard_scalers.pkl` - Feature normalization scalers
- `requirements.txt` - Python dependencies

## Setup

### 1. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip -y

# Install PyTorch for ARM (Raspberry Pi)
# Check latest version at: https://pytorch.org/get-started/locally/
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip3 install -r requirements.txt
```

### 2. Configure Raspberry Pi Hotspot

The ESP32 connects to a WiFi hotspot on the Raspberry Pi. Configure the hotspot:

```bash
# Install hostapd and dnsmasq
sudo apt install hostapd dnsmasq -y

# Configure hotspot (example for hostapd)
# Create /etc/hostapd/hostapd.conf with:
#   ssid=Pi-Hotspot
#   wpa_passphrase=MacEmgConnect
#   interface=wlan0
#   driver=nl80211
#   channel=7
#   hw_mode=g
#   wpa=2
#   wpa_key_mgmt=WPA-PSK

# Start hotspot
sudo systemctl enable hostapd
sudo systemctl start hostapd
```

**Note:** The hotspot should have IP `192.168.4.1` which matches ESP32 configuration.

### 3. Run Inference

```bash
# Basic usage (from deployment-3 directory)
python3 inference.py

# With custom options
python3 inference.py --port 5000 --device cpu

# See all options
python3 inference.py --help
```

## Usage

### Starting the System

1. Start the Raspberry Pi hotspot (if not auto-started)
2. Run the inference script:
   ```bash
   python3 inference.py
   ```

3. The system will:
   - Load the model and configuration
   - Start the TCP server on port 5000
   - Wait for ESP32 connection
   - Begin processing data and making predictions

### Terminal Output

The system provides real-time terminal logging:

```
======================================================================
🚀 Initializing EMG Gesture Classification System
======================================================================
📂 Model directory: ./
✅ Model loaded successfully
   Parameters: 15,370
   Classes: ['REST', 'FIST']

======================================================================
🎯 Starting Real-Time Inference
======================================================================
   TCP Port: 5000
   Device: cpu
   Window size: 100 samples (200ms)
   Sequence length: 8 windows

──────────────────────────────────────────────────────────────────────
📊 Waiting for ESP32 connection...
──────────────────────────────────────────────────────────────────────

📡 TCP Server started on 0.0.0.0:5000
   Waiting for ESP32 connection...
✅ ESP32 connected from ('192.168.4.2', 54321)

[14:23:45.123] 🛑 REST  Confidence: 87.50% ██████████  | REST: 87.50%  FIST: 12.50%
[14:23:45.373] ✊ FIST  Confidence: 92.30% ██████████  | REST: 7.70%  FIST: 92.30%
```

### System Statistics

Every 5 seconds, the system prints statistics:

```
──────────────────────────────────────────────────────────────────────
📊 System Statistics
──────────────────────────────────────────────────────────────────────
   Runtime: 45.2s
   ESP32 Status: ✅ Connected
   Batches received: 904
   Total samples: 45,200
   Windows processed: 1,808
   Sequences processed: 72
   Last batch: 0.05s ago
   Recent predictions (last 10): REST=3, FIST=7
   Average confidence: 85.20%
──────────────────────────────────────────────────────────────────────
```

## Configuration

### Model Parameters

Model configuration is loaded from `deployment_config.json`:

- `window_size`: 100 samples (200ms @ 500Hz)
- `stride`: 25 samples (50ms, 50% overlap)
- `sequence_length`: 8 windows
- `sampling_rate`: 500 Hz (EMG)
- `imu_sampling_rate`: 100 Hz

### ESP32 Connection

The ESP32 should connect to:
- **SSID**: `Pi-Hotspot`
- **Password**: `MacEmgConnect`
- **Server IP**: `192.168.4.1`
- **Port**: `5000`

## Features

### Real-Time Processing

- **Sliding Window**: 100-sample windows with 50% overlap
- **Sequence Generation**: 8-window sequences for LSTM temporal modeling
- **Feature Extraction**: RMS/LMS envelope, 3 bandpowers, 4 IMU stats
- **Normalization**: Global StandardScaler (matches training)

### Terminal Logging

- Real-time prediction display with color coding
- Confidence bars and probability breakdowns
- System statistics every 5 seconds
- Connection status monitoring

## Troubleshooting

### ESP32 Not Connecting

1. Check hotspot is running:
   ```bash
   sudo systemctl status hostapd
   ```

2. Verify IP address:
   ```bash
   ip addr show wlan0
   # Should show 192.168.4.1
   ```

3. Check firewall:
   ```bash
   sudo ufw allow 5000/tcp
   ```

### Model Loading Errors

1. Verify all files are present:
   ```bash
   ls -la deployment-3/
   # Should show: emg_lstm_model.pt, deployment_config.json, standard_scalers.pkl
   ```

2. Check file permissions:
   ```bash
   chmod +r deployment-3/*.pt deployment-3/*.json deployment-3/*.pkl
   ```

### Low Prediction Accuracy

- Ensure EMG sensor is properly attached
- Check signal quality (monitor terminal for consistent batches)
- Verify normalization is working (check scalers loaded correctly)

## Performance

### Expected Performance

- **Processing latency**: ~10-20ms per sequence
- **Throughput**: ~10-50 sequences/second (depending on CPU)
- **Memory usage**: ~100-200 MB (model + buffers)

### Optimization Tips

1. Use CPU affinity for better performance:
   ```bash
   taskset -c 0-3 python3 inference.py
   ```

2. Disable unnecessary services to free CPU
3. Use a Raspberry Pi 4 (recommended) for better performance

## REST-Baseline Logic (Future)

REST-baseline logic will be added in a future update. When enabled, the system will:

1. Collect REST windows during session initialization (2 seconds)
2. Compute per-session baseline statistics
3. Normalize incoming windows relative to baseline
4. Improve robustness to day-to-day changes

## Support

For issues or questions, check:
- Model configuration in `deployment_config.json`
- System logs in terminal output
- ESP32 serial monitor for connection issues

## License

See main project license.

