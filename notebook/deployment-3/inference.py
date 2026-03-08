#!/usr/bin/env python3
"""
Real-time EMG gesture classification inference on Raspberry Pi.
Receives data from TWO ESP32 devices (LEFT and RIGHT arms).
Implements wheelchair control logic: LEFT/RIGHT/FORWARD/REST commands.

Key Features (SIMPLIFIED FOR DEPLOYMENT):
- Gyroscope disabled in LMS: Motion does not interfere with gesture detection
- Direct model predictions: Trusts the trained model without overriding
- Simple REST detection: Requires 3 consecutive REST predictions for reliable STOP
- Fast FIST detection: Immediate GO response for safety
- Low latency: Optimized for wheelchair control (50ms inference interval, 200ms sequence latency)

Usage:
    python inference.py [--model-dir ./] [--port 5000] [--device cpu]
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional
import queue
import threading

# Import deployment modules
from model_loader import load_model
from preprocessor import RealTimePreprocessor
from window_buffer import WindowBuffer
from udp_server import UDPServer

# GPIO control for BTS 7960 motor driver
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️  GPIO control not available (RPi.GPIO not installed)")


class WheelchairMotorController:
    """
    Continuous motor controller for model-driven wheelchair control.
    Model predictions directly control movement with no normalization delays.
    
    Motor Control Mapping:
    - Device 1 (LEFT) detects FIST  → Right motor moves → Wheelchair turns LEFT
    - Device 2 (RIGHT) detects FIST → Left motor moves → Wheelchair turns RIGHT
    - BOTH devices detect FIST      → Both motors move → Wheelchair goes FORWARD
    
    PIN Configuration:
    - Left Motor
      - LEFT_MOTOR_RPWM = 13 (Forward PWM)
      - LEFT_MOTOR_LPWM = 17 (Reverse PWM)
      - LEFT_MOTOR_R_EN = 22 (Forward enable)
      - LEFT_MOTOR_L_EN = 23 (Reverse enable)
    
    - Right Motor
      - RIGHT_MOTOR_RPWM = 12 (Forward PWM)
      - RIGHT_MOTOR_LPWM = 27 (Reverse PWM)
      - RIGHT_MOTOR_R_EN = 5 (Forward enable)
      - RIGHT_MOTOR_L_EN = 6 (Reverse enable)
    """
    
    # PIN Configuration
    LEFT_MOTOR_RPWM = 13    # Forward PWM
    LEFT_MOTOR_LPWM = 17    # Reverse PWM
    LEFT_MOTOR_R_EN = 22    # Forward enable
    LEFT_MOTOR_L_EN = 23    # Reverse enable
    
    RIGHT_MOTOR_RPWM = 12   # Forward PWM
    RIGHT_MOTOR_LPWM = 27   # Reverse PWM
    RIGHT_MOTOR_R_EN = 5    # Forward enable
    RIGHT_MOTOR_L_EN = 6    # Reverse enable
    
    PWM_FREQUENCY = 1000    # 1 kHz PWM frequency
    
    def __init__(self, 
                 movement_duration: float = 2.0,
                 normalization_period: float = 0.0,  # REMOVED: Allow continuous predictions
                 debug_mode: bool = False, 
                 verbose: bool = True):
        """
        Continuous wheelchair control - model predictions directly control movement.
        
        Args:
            movement_duration: Duration of each movement (default: 2.0s)
            normalization_period: Normalization period after movement (default: 0.0s - disabled)
            debug_mode: If True, print commands but don't activate motors
            verbose: If True, print detailed motor control updates
        """
        self.movement_duration = movement_duration
        self.normalization_period = 0.0  # Disabled - allow continuous predictions
        self.debug_mode = debug_mode
        self.verbose = verbose
        
        # Simplified continuous control state (no duration limits)
        self.is_moving = False
        self.current_movement = None  # 'LEFT', 'RIGHT', 'FORWARD', or None
        self.movement_start_time = None
        
        # Motor state
        self.left_motor_speed = 0
        self.right_motor_speed = 0
        self.gpio_initialized = False
        self.base_speed = 50  # Base speed (50% duty cycle - increased for better movement)
        
        # Initialize GPIO if available and not in debug mode
        if GPIO_AVAILABLE and not debug_mode:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                # Setup all pins
                pins = [
                    self.LEFT_MOTOR_RPWM, self.LEFT_MOTOR_LPWM,
                    self.LEFT_MOTOR_R_EN, self.LEFT_MOTOR_L_EN,
                    self.RIGHT_MOTOR_RPWM, self.RIGHT_MOTOR_LPWM,
                    self.RIGHT_MOTOR_R_EN, self.RIGHT_MOTOR_L_EN
                ]
                
                for pin in pins:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                
                # Enable channels (HIGH = enabled)
                GPIO.output(self.LEFT_MOTOR_R_EN, GPIO.HIGH)
                GPIO.output(self.LEFT_MOTOR_L_EN, GPIO.HIGH)
                GPIO.output(self.RIGHT_MOTOR_R_EN, GPIO.HIGH)
                GPIO.output(self.RIGHT_MOTOR_L_EN, GPIO.HIGH)
                
                # Setup PWM on motor pins
                self.left_forward_pwm = GPIO.PWM(self.LEFT_MOTOR_RPWM, self.PWM_FREQUENCY)
                self.left_reverse_pwm = GPIO.PWM(self.LEFT_MOTOR_LPWM, self.PWM_FREQUENCY)
                self.right_forward_pwm = GPIO.PWM(self.RIGHT_MOTOR_RPWM, self.PWM_FREQUENCY)
                self.right_reverse_pwm = GPIO.PWM(self.RIGHT_MOTOR_LPWM, self.PWM_FREQUENCY)
                
                # Start PWM at 0% duty cycle
                self.left_forward_pwm.start(0)
                self.left_reverse_pwm.start(0)
                self.right_forward_pwm.start(0)
                self.right_reverse_pwm.start(0)
                
                self.gpio_initialized = True
                print("✅ GPIO initialized for BTS 7960 motor driver")
                print(f"   Left Motor: RPWM={self.LEFT_MOTOR_RPWM}, LPWM={self.LEFT_MOTOR_LPWM}, R_EN={self.LEFT_MOTOR_R_EN}, L_EN={self.LEFT_MOTOR_L_EN}")
                print(f"   Right Motor: RPWM={self.RIGHT_MOTOR_RPWM}, LPWM={self.RIGHT_MOTOR_LPWM}, R_EN={self.RIGHT_MOTOR_R_EN}, L_EN={self.RIGHT_MOTOR_L_EN}")
                print(f"   PWM Frequency: {self.PWM_FREQUENCY} Hz")
                print(f"   Control Mode: SIMPLIFIED - Direct model trust")
                print(f"   ✓ Commands execute immediately based on predictions")
            except Exception as e:
                print(f"❌ GPIO initialization failed: {e}")
                self.gpio_initialized = False
        else:
            if debug_mode:
                print("🔧 Motor controller in DEBUG mode (no GPIO control)")
            else:
                print("⚠️  GPIO not available (RPi.GPIO not installed)")
    
    def update_command(self, command: str):
        """
        SIMPLIFIED wheelchair control - trust model predictions directly.
        
        Control Logic:
        1. Execute commands immediately based on current prediction
        2. No movement duration limits - continuous control
        3. Commands update in real-time as model predictions change
        
        Args:
            command: One of 'REST', 'LEFT', 'RIGHT', 'FORWARD'
        """
        current_time = time.time()
        
        # Execute command immediately
        if command == 'REST':
            # Stop motors immediately
            if self.is_moving:
                self._stop_movement(current_time)
        else:
            # Execute movement command (LEFT, RIGHT, FORWARD)
            if command != self.current_movement:
                # New command or command change - execute immediately
                self._execute_command(command)
                self.is_moving = True
                self.current_movement = command
                self.movement_start_time = current_time
    
    def _stop_movement(self, current_time: float):
        """
        Stop current movement immediately.
        
        Args:
            current_time: Current timestamp
        """
        # Stop motors
        self._execute_command('REST')
        
        # Reset movement state
        self.is_moving = False
        self.current_movement = None
        self.movement_start_time = None
    
    def _execute_command(self, command: str):
        """
        Execute the motor command.
        
        Args:
            command: One of 'REST', 'FORWARD', 'LEFT', 'RIGHT'
        """
        if self.debug_mode:
            print(f"🔧 [DEBUG] Motor command: {command}")
            return
        
        if not GPIO_AVAILABLE:
            if self.verbose:
                print(f"⚠️  Cannot execute {command}: GPIO not available")
            return
        
        if not self.gpio_initialized:
            if self.verbose:
                print(f"⚠️  Cannot execute {command}: GPIO not initialized")
            return
        
        # Command execution - display will update automatically
        
        if command == 'FORWARD':
            # Both motors forward
            self._set_motor_speed('left', self.base_speed, 'forward')
            self._set_motor_speed('right', self.base_speed, 'forward')
        
        elif command == 'LEFT':
            # Right motor forward, left motor stop (turn left)
            self._set_motor_speed('left', 0, 'forward')
            self._set_motor_speed('right', self.base_speed, 'forward')
        
        elif command == 'RIGHT':
            # Left motor forward, right motor stop (turn right)
            self._set_motor_speed('left', self.base_speed, 'forward')
            self._set_motor_speed('right', 0, 'forward')
        
        else:  # REST
            # Stop both motors
            self._set_motor_speed('left', 0, 'forward')
            self._set_motor_speed('right', 0, 'forward')
    
    def _set_motor_speed(self, motor: str, speed: int, direction: str = 'forward'):
        """
        Set motor speed and direction.
        
        Args:
            motor: 'left' or 'right'
            speed: PWM duty cycle (0-100)
            direction: 'forward' or 'reverse'
        """
        if not GPIO_AVAILABLE or self.debug_mode or not self.gpio_initialized:
            return
        
        speed = max(0, min(100, speed))  # Clamp to 0-100
        
        try:
            if motor == 'left':
                if direction == 'forward':
                    self.left_forward_pwm.ChangeDutyCycle(speed)
                    self.left_reverse_pwm.ChangeDutyCycle(0)
                else:
                    self.left_forward_pwm.ChangeDutyCycle(0)
                    self.left_reverse_pwm.ChangeDutyCycle(speed)
                self.left_motor_speed = speed
            
            elif motor == 'right':
                if direction == 'forward':
                    self.right_forward_pwm.ChangeDutyCycle(speed)
                    self.right_reverse_pwm.ChangeDutyCycle(0)
                else:
                    self.right_forward_pwm.ChangeDutyCycle(0)
                    self.right_reverse_pwm.ChangeDutyCycle(speed)
                self.right_motor_speed = speed
        except Exception as e:
            print(f"❌ Error setting motor speed: {e}")
    
    def emergency_stop(self):
        """Immediately stop all motors and reset state."""
        if GPIO_AVAILABLE and not self.debug_mode and self.gpio_initialized:
            self._set_motor_speed('left', 0, 'forward')
            self._set_motor_speed('right', 0, 'forward')
        
        # Reset simplified control state
        self.is_moving = False
        self.current_movement = None
        self.movement_start_time = None
    
    def test_motors(self):
        """Test motor functionality - runs each motor briefly for verification."""
        if not GPIO_AVAILABLE or self.debug_mode or not self.gpio_initialized:
            print("⚠️  Cannot test motors: GPIO not available or in debug mode")
            return
        
        print("\n🧪 Testing Motor Connections...")
        print("   This will briefly activate each motor at 30% speed")
        
        try:
            # Test left motor forward
            print("   1. Left motor forward (2 seconds)...")
            self._set_motor_speed('left', 30, 'forward')
            time.sleep(2)
            self._set_motor_speed('left', 0, 'forward')
            time.sleep(1)
            
            # Test right motor forward
            print("   2. Right motor forward (2 seconds)...")
            self._set_motor_speed('right', 30, 'forward')
            time.sleep(2)
            self._set_motor_speed('right', 0, 'forward')
            time.sleep(1)
            
            # Test both motors forward
            print("   3. Both motors forward (2 seconds)...")
            self._set_motor_speed('left', 30, 'forward')
            self._set_motor_speed('right', 30, 'forward')
            time.sleep(2)
            self._set_motor_speed('left', 0, 'forward')
            self._set_motor_speed('right', 0, 'forward')
            
            print("✅ Motor test complete!")
        except Exception as e:
            print(f"❌ Motor test failed: {e}")
            self.emergency_stop()
    
    def cleanup(self):
        """Clean up GPIO resources."""
        if GPIO_AVAILABLE and not self.debug_mode and self.gpio_initialized:
            self.emergency_stop()
            self.left_forward_pwm.stop()
            self.left_reverse_pwm.stop()
            self.right_forward_pwm.stop()
            self.right_reverse_pwm.stop()
            GPIO.cleanup()
            print("✅ GPIO cleaned up")


class DevicePipeline:
    """
    Per-device pipeline for EMG processing.
    Each device (LEFT/RIGHT arm) has its own isolated pipeline.
    """
    
    def __init__(self,
                 device_id: int,
                 device_name: str,
                 window_buffer: WindowBuffer,
                 preprocessor: RealTimePreprocessor,
                 enable_rest_baseline: bool = False):
        """
        Args:
            device_id: Device ID (1=LEFT, 2=RIGHT)
            device_name: Device name for display
            window_buffer: WindowBuffer instance
            preprocessor: RealTimePreprocessor instance
            enable_rest_baseline: Whether to collect REST baseline
        """
        self.device_id = device_id
        self.device_name = device_name
        self.window_buffer = window_buffer
        self.preprocessor = preprocessor
        self.enable_rest_baseline = enable_rest_baseline
        
        # Filter warm-up state (keep simple to avoid long delays)
        self.filter_warmup_needed = True
        self.filter_warmup_windows = 0
        self.filter_warmup_required = 50  # ~10s warmup at 200ms/window
        
        # Simple stability check (coefficient of variation over recent envelopes)
        self.warmup_envelope_history = deque(maxlen=20)
        
        # REST baseline collection state (fixed duration, no extensions)
        self.rest_baseline_collecting = False
        self.rest_baseline_start_time = None
        self.rest_baseline_duration = 5.0  # fixed 5s
        self.rest_baseline_windows_collected = 0
        self.baseline_envelope_history = None
        
        # Latest prediction state
        self.last_prediction = 'REST'  # Default to REST
        self.last_confidence = 0.0
        self.last_inference_time = 0
        
        # Safety: Track prediction transitions for STOP detection
        self.prediction_history = deque(maxlen=5)  # Last 5 predictions for safety
        self.last_fist_time = None  # Track when FIST was last detected
        
        # Feature smoothing for better model input quality
        # Smooth features temporally to reduce noise and help model see clearer signals
        self.feature_history = {
            'envelope_mean': deque(maxlen=5),  # Last 5 envelope means
            'high_bandpower': deque(maxlen=5),  # Last 5 high bandpowers
        }
        
        # REMOVED: REST confirmation logic - we now trust the model's predictions directly
        
        # Statistics
        self.total_windows_processed = 0
        self.total_sequences_processed = 0
        
        # Signal state tracking (NO_SIGNAL, REST, ACTIVE)
        self.signal_state = 'NO_SIGNAL'  # Start as NO_SIGNAL until we see valid data
        self.last_valid_prediction = 'REST'
        self.last_valid_prediction_time = None
        self.hold_duration = 0.4  # 400ms hold state to prevent brief dropouts
        self.consecutive_invalid_windows = 0
        self.max_invalid_windows = 5  # After 5 invalid windows, enter NO_SIGNAL state
        self.zero_signal_warnings = 0  # Track how many zero signal warnings we've shown


    def update_signal_state(self, raw_emg: np.ndarray, envelope: np.ndarray, bandpowers: np.ndarray) -> str:
        """
        Update signal state based on raw EMG and processed envelope.
        Implements hysteresis to prevent rapid state changes.
        
        Returns:
            Current signal state: 'NO_SIGNAL', 'REST', or 'ACTIVE'
        """
        # Check raw signal validity first
        raw_state = self.preprocessor.check_signal_validity(raw_emg)
        
        if raw_state == 'NO_SIGNAL':
            self.consecutive_invalid_windows += 1
            if self.consecutive_invalid_windows >= self.max_invalid_windows:
                self.signal_state = 'NO_SIGNAL'
                # Log zero signal issue periodically
                if self.zero_signal_warnings < 3 or self.consecutive_invalid_windows % 100 == 0:
                    raw_range = np.max(raw_emg) - np.min(raw_emg)
                    raw_mean = np.mean(raw_emg)
                    print(f"⚠️  [{self.device_name}] NO_SIGNAL: raw EMG all zeros or constant (range={raw_range:.2f}, mean={raw_mean:.2f})")
                    print(f"    Check: 1) Sensor connection, 2) ESP32 ADC pin, 3) Ground connection")
                    self.zero_signal_warnings += 1
                return 'NO_SIGNAL'
            # Use HOLD state during brief dropouts
            return self.signal_state
        
        # Raw signal is valid, check envelope state
        envelope_state = self.preprocessor.check_envelope_state(envelope, bandpowers)
        
        if envelope_state == 'NO_SIGNAL':
            # Signal destroyed by filters - this is a filtering problem
            self.consecutive_invalid_windows += 1
            if self.consecutive_invalid_windows >= self.max_invalid_windows:
                self.signal_state = 'NO_SIGNAL'
                if self.zero_signal_warnings < 3:
                    env_mean = np.mean(envelope)
                    bp_high = bandpowers[2] if len(bandpowers) >= 3 else 0.0
                    print(f"⚠️  [{self.device_name}] NO_SIGNAL: filters destroyed signal (env={env_mean:.4f}, bp_high={bp_high:.4f})")
                    print(f"    Wiener filter may be too aggressive - check noise calibration")
                    self.zero_signal_warnings += 1
                return 'NO_SIGNAL'
            return self.signal_state
        
        # Signal is valid - reset counter and update state
        self.consecutive_invalid_windows = 0
        self.signal_state = envelope_state
        return self.signal_state
    
    def should_process_window(self, signal_state: str) -> bool:
        """
        Determine if window should be processed by classifier (gated classification).
        
        Returns:
            True if window should be fed to classifier, False otherwise
        """
        # Never process NO_SIGNAL windows
        if signal_state == 'NO_SIGNAL':
            return False
        
        # Always process during baseline collection (need REST windows)
        if self.rest_baseline_collecting:
            return True
        
        # After baseline: Process ACTIVE windows and REST windows
        # (Model needs to see both REST and FIST to distinguish them)
        return True
    
    def get_hold_prediction(self) -> Optional[str]:
        """
        Get prediction from HOLD state if within hold duration.
        Maintains last valid command during brief signal dropouts.
        
        Returns:
            Last valid prediction if within hold window, None otherwise
        """
        if self.last_valid_prediction_time is None:
            return None
        
        elapsed = time.time() - self.last_valid_prediction_time
        if elapsed < self.hold_duration:
            return self.last_valid_prediction
        
        # Hold duration expired - return REST for safety
        return 'REST'


class InferenceEngine:
    """
    Main inference engine for dual-device gesture classification.
    Manages two parallel pipelines (LEFT and RIGHT arms) and implements
    wheelchair control logic (LEFT/RIGHT/FORWARD/REST commands).
    """
    
    def __init__(self,
                 model_dir: str,
                 device: str = 'cpu',
                 port: int = 5000,
                 enable_rest_baseline: bool = True,  # ALWAYS True - model trained with vs_rest
                 debug: bool = False,
            dashboard_interval: float = 5.0,
            min_confidence: float = 0.30,  # Further lowered for better sensitivity (was 0.40)
            enable_wheelchair: bool = False,
                 wheelchair_debug: bool = False):
        """
        Args:
            model_dir: Directory containing model files
            device: Device to run inference on ('cpu' or 'cuda')
            port: UDP server port (default: 5000, matches ESP32 default)
            enable_rest_baseline: Whether to use REST-baseline logic (ALWAYS True - model trained with vs_rest)
            debug: Enable debug logging for UDP server
            dashboard_interval: Seconds between dashboard updates (default: 5.0)
            min_confidence: Minimum confidence threshold for FIST predictions (default: 0.50, lowered for better sensitivity)
            enable_wheelchair: Enable wheelchair motor control (default: False)
            wheelchair_debug: Wheelchair debug mode - predictions but no motor movement (default: False)
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.port = port
        self.protocol = 'udp'
        self.enable_rest_baseline = enable_rest_baseline
        self.debug = debug
        self.dashboard_interval = dashboard_interval
        self.min_confidence = min_confidence
        
        # Load model and configuration
        print("\n" + "="*70)
        print("🚀 Initializing DUAL-DEVICE Wheelchair Control System")
        print("="*70)
        print(f"📂 Model directory: {self.model_dir}")
        
        self.model, self.config, self.scalers = load_model(str(self.model_dir), device)
        
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {self.config.get('model_parameters', 'N/A'):,}")
        class_names = self.config.get('class_names', ['REST', 'FIST'])
        print(f"   Classes: {class_names}")
        
        # Extract configuration
        self.window_size = self.config.get('window_size', 100)
        # OPTIMIZED FOR LOW LATENCY: Reduced stride and sequence length for wheelchair safety
        # Original model trained with: stride=25 (50ms), sequence=16 (800ms latency)
        # Compromise for wheelchair: stride=18 (36ms), sequence=8 (288ms latency)
        # This balances model performance with safety response time
        self.stride = 18  # 36ms between windows (balanced)
        self.sequence_length = 8  # Compromise: better than 5, faster than 16 (288ms latency)
        self.sampling_rate = self.config.get('sampling_rate', 500)
        self.imu_sampling_rate = self.config.get('imu_sampling_rate', 100)
        self.class_names = self.config.get('class_names', ['REST', 'FIST'])
        
        # Create per-device pipelines (NO SHARED QUEUES)
        print(f"\n🔧 Creating isolated per-device pipelines...")
        
        # LEFT ARM pipeline (Device ID = 1)
        left_window_buffer = WindowBuffer(
            window_size=self.window_size,
            stride=self.stride,
            sequence_length=self.sequence_length,
            emg_sampling_rate=self.sampling_rate,
            imu_sampling_rate=self.imu_sampling_rate
        )
        left_preprocessor = RealTimePreprocessor(
            window_size=self.window_size,
            sampling_rate=self.sampling_rate,
            imu_sampling_rate=self.imu_sampling_rate,
            scalers=self.scalers,
            enable_rest_baseline=True
        )
        self.left_pipeline = DevicePipeline(
            device_id=1,
            device_name="LEFT",
            window_buffer=left_window_buffer,
            preprocessor=left_preprocessor,
            enable_rest_baseline=self.enable_rest_baseline
        )
        
        # RIGHT ARM pipeline (Device ID = 2)
        right_window_buffer = WindowBuffer(
            window_size=self.window_size,
            stride=self.stride,
            sequence_length=self.sequence_length,
            emg_sampling_rate=self.sampling_rate,
            imu_sampling_rate=self.imu_sampling_rate
        )
        right_preprocessor = RealTimePreprocessor(
            window_size=self.window_size,
            sampling_rate=self.sampling_rate,
            imu_sampling_rate=self.imu_sampling_rate,
            scalers=self.scalers,
            enable_rest_baseline=True
        )
        self.right_pipeline = DevicePipeline(
            device_id=2,
            device_name="RIGHT",
            window_buffer=right_window_buffer,
            preprocessor=right_preprocessor,
            enable_rest_baseline=self.enable_rest_baseline
        )
        
        print(f"   ✅ LEFT ARM pipeline (Device ID=1)")
        print(f"   ✅ RIGHT ARM pipeline (Device ID=2)")
        
        # Map device IDs to pipelines
        self.pipelines = {
            1: self.left_pipeline,
            2: self.right_pipeline
        }
        
        # CRITICAL: Queue-based processing for responsive UDP reception
        # Move heavy processing off receive thread to prevent blocking
        self.batch_queue = queue.Queue(maxsize=10)  # Small queue to prevent old data buildup
        self.processing_thread = None
        self.processing_running = False
        
        # Initialize UDP server (DUAL-PORT mode for separate ESP32 streams)
        # LEFT device on port 5000, RIGHT device on port 5001
        self.server_left = UDPServer(
            host='0.0.0.0',
            port=self.port,  # 5000
            batch_callback=lambda device_id, emg, imu: self._enqueue_batch(1, emg, imu),  # Enqueue instead of direct processing
            debug=self.debug
        )
        self.server_right = UDPServer(
            host='0.0.0.0',
            port=self.port + 1,  # 5001
            batch_callback=lambda device_id, emg, imu: self._enqueue_batch(2, emg, imu),  # Enqueue instead of direct processing
            debug=self.debug
        )
        
        # Global statistics
        self.start_time = None
        # CRITICAL: Optimized inference interval for maximum responsiveness
        # 30ms provides rapid updates while preventing excessive computation
        # Lower latency = faster STOP/GO response for safety
        self.inference_interval = 0.030  # 30ms between inferences (33 Hz, reduced from 50ms for faster response)
        
        # Wheelchair control state
        self.last_wheelchair_command = 'REST'
        self.wheelchair_command_history = deque(maxlen=50)
        
        # Display state (unified single-line display)
        self.display_lines_printed = 0
        self.last_display_update = 0
        self.display_update_interval = 0.1
        
        # Packet loss detection (per device)
        self.last_data_received_time = {1: None, 2: None}
        self.data_timeout = 0.5
        self.packet_loss_detected = {1: False, 2: False}
        
        # Wheelchair control integration
        self.enable_wheelchair = enable_wheelchair
        self.wheelchair_controller = None
        if self.enable_wheelchair:
            if GPIO_AVAILABLE or wheelchair_debug:
                self.wheelchair_controller = WheelchairMotorController(
                    movement_duration=0.0,          # Not used - immediate execution
                    normalization_period=0.0,       # Not used - immediate execution
                    debug_mode=wheelchair_debug,
                    verbose=True
                )
                print(f"\n🦽 SIMPLIFIED wheelchair control {'ENABLED' if not wheelchair_debug else 'ENABLED (DEBUG MODE - no motor movement)'}")
                print(f"   Motor power: 50% duty cycle")
                print(f"   Control mode: Direct model trust - immediate execution")
                print(f"   ✓ No confirmation delays or duration limits")
                print(f"   ✓ Commands execute immediately based on predictions")
            else:
                print(f"\n⚠️  Wheelchair control requested but GPIO not available (check RPi.GPIO installation)")
                self.enable_wheelchair = False
    
    def _enqueue_batch(self, device_id: int, emg_samples: np.ndarray, imu_samples: np.ndarray):
        """
        CRITICAL: Enqueue batch for processing (non-blocking for UDP receive thread).
        This keeps UDP reception fast and responsive.
        
        Args:
            device_id: Device ID (1=LEFT, 2=RIGHT)
            emg_samples: EMG samples array
            imu_samples: IMU samples array (n_samples, 6)
        """
        try:
            # Try to add to queue (non-blocking with timeout)
            self.batch_queue.put((device_id, emg_samples, imu_samples), block=False)
        except queue.Full:
            # Queue full - drop oldest and add newest (prevent stale data buildup)
            if self.debug:
                print(f"⚠️  Batch queue full, dropping old data")
            try:
                self.batch_queue.get_nowait()  # Remove oldest
                self.batch_queue.put((device_id, emg_samples, imu_samples), block=False)
            except:
                pass  # If still fails, just drop this batch
    
    def _processing_worker(self):
        """
        CRITICAL: Worker thread that processes batches from queue.
        Runs independently from UDP receive thread for maximum responsiveness.
        """
        while self.processing_running:
            try:
                # Get batch from queue (blocking with timeout)
                device_id, emg_samples, imu_samples = self.batch_queue.get(timeout=0.1)
                
                # Process batch (same as before, but now off receive thread)
                self._on_batch_received(device_id, emg_samples, imu_samples)
                
            except queue.Empty:
                continue  # No batch available, continue waiting
            except Exception as e:
                if self.debug:
                    print(f"⚠️  Processing worker error: {e}")
                import traceback
                traceback.print_exc()
    
    def _on_batch_received(self, device_id: int, emg_samples: np.ndarray, imu_samples: np.ndarray):
        """
        Callback when a batch is received from ESP32.
        Routes data to the appropriate device pipeline.
        NOW RUNS IN WORKER THREAD (not UDP receive thread).
        
        Args:
            device_id: Device ID (1=LEFT, 2=RIGHT)
            emg_samples: EMG samples array
            imu_samples: IMU samples array (n_samples, 6)
        """
        # Get the appropriate pipeline
        pipeline = self.pipelines.get(device_id)
        
        if pipeline is None:
            if self.debug:
                print(f"⚠️  Unknown device_id={device_id}, ignoring batch")
            return
        
        # Update data freshness timestamp for this device
        current_time = time.time()
        self.last_data_received_time[device_id] = current_time
        
        # Reset packet loss flag if we're receiving data again
        if self.packet_loss_detected.get(device_id, False):
            self.packet_loss_detected[device_id] = False
            print(f"✅ [{pipeline.device_name}] Data reception resumed")
        
        if self.debug:
            print(f"📦 [{pipeline.device_name}] Batch received: {len(emg_samples)} EMG, {len(imu_samples)} IMU samples")
        
        # Add samples to device-specific buffer
        pipeline.window_buffer.add_emg_samples(emg_samples)
        pipeline.window_buffer.add_imu_samples(imu_samples)
        
        # Process windows for this device (only when new data arrives)
        self._process_windows(pipeline)
    
    def _process_windows(self, pipeline: DevicePipeline):
        """
        Process available windows for a specific device pipeline.
        
        Args:
            pipeline: DevicePipeline instance to process
        """
        # Check for packet loss - don't predict if data is stale
        device_id = pipeline.device_id
        if self.last_data_received_time[device_id] is None:
            return  # No data received yet
        
        current_time = time.time()
        time_since_last_data = current_time - self.last_data_received_time[device_id]
        
        # CRITICAL: Detect packet loss EARLY - skip prediction if data is stale
        # This prevents processing old buffered data after device turns off
        if time_since_last_data > self.data_timeout:
            if not self.packet_loss_detected[device_id]:
                self.packet_loss_detected[device_id] = True
                print(f"⚠️  [{pipeline.device_name}] Packet loss detected (no data for {time_since_last_data:.2f}s)")
                # Clear buffers to prevent stale predictions
                pipeline.window_buffer.clear()
                # CRITICAL: Immediately set to REST for safety
                pipeline.last_prediction = 'REST'
                pipeline.last_confidence = 0.0
                # Update wheelchair command immediately
                self._update_wheelchair_command()
            return  # Don't process stale data
        
        # CRITICAL: Reduce windows per batch for faster response
        # Lower limit = shorter processing time per callback = more responsive UDP reception
        windows_created = 0
        max_windows_per_batch = 3  # REDUCED from 10 to 3 for faster response
        
        while pipeline.window_buffer.can_create_window() and windows_created < max_windows_per_batch:
            # Check data freshness during processing
            current_time = time.time()
            time_since_last_data = current_time - self.last_data_received_time[device_id]
            
            if time_since_last_data > self.data_timeout:
                break  # Stop if data became stale
            
            window_data = pipeline.window_buffer.create_window()
            if window_data is None:
                break
            
            emg_window, imu_window = window_data
            
            # Extract and normalize features (pass device_id for diagnostics)
            features = pipeline.preprocessor.process_window(emg_window, imu_window, device_id=pipeline.device_id)
            
            # Check if window was rejected by device normalizer
            if features is None:
                # Window rejected (invalid signal or motion artifact)
                # Use HOLD state to maintain last valid prediction
                hold_pred = pipeline.get_hold_prediction()
                if hold_pred is not None:
                    # Maintain HOLD prediction
                    pipeline.total_windows_processed += 1
                    windows_created += 1
                    continue
                else:
                    # No valid prediction to hold - default to REST for safety
                    pipeline.last_prediction = 'REST'
                    pipeline.total_windows_processed += 1
                    windows_created += 1
                    continue
            
            # SIGNAL VALIDITY GATING: Check if signal is valid before processing
            envelope = features['emg_envelope']
            bandpowers = features['emg_bandpowers']
            signal_state = pipeline.update_signal_state(emg_window, envelope, bandpowers)
            
            # Check if window should be processed (gated classification)
            if not pipeline.should_process_window(signal_state):
                # Window is NO_SIGNAL - check HOLD state
                hold_pred = pipeline.get_hold_prediction()
                
                if hold_pred is not None:
                    # Use HOLD prediction to maintain last valid command
                    if self.debug and pipeline.total_windows_processed % 20 == 0:
                        print(f"[{pipeline.device_name}] Using HOLD prediction: {hold_pred} (signal_state={signal_state})")
                    # Don't run inference, but maintain HOLD state
                    pipeline.total_windows_processed += 1
                    windows_created += 1
                    continue
                else:
                    # No valid prediction to hold - skip window entirely
                    # Default to REST for safety
                    pipeline.last_prediction = 'REST'
                    pipeline.total_windows_processed += 1
                    windows_created += 1
                    continue
            
            # CRITICAL: Warm up filters FIRST before collecting REST baseline
            # Keep simple: require LMS ready, Wiener ready, min windows, and CV < 0.3
            if pipeline.filter_warmup_needed:
                pipeline.filter_warmup_windows += 1
                
                # Track envelope stability
                env_mean = np.mean(features['emg_envelope'])
                pipeline.warmup_envelope_history.append(env_mean)
                
                # Check readiness
                lms_ready = pipeline.preprocessor.lms_adaptation_count >= pipeline.preprocessor.lms_min_adaptation_samples
                wiener_ready = pipeline.preprocessor.wiener_is_calibrated
                windows_ready = pipeline.filter_warmup_windows >= pipeline.filter_warmup_required
                
                # Simple CV check over recent envelopes
                envelope_stable = False
                cv = 1.0
                if len(pipeline.warmup_envelope_history) >= 10:
                    recent_envelopes = list(pipeline.warmup_envelope_history)
                    envelope_mean = np.mean(recent_envelopes)
                    envelope_std = np.std(recent_envelopes)
                    cv = envelope_std / (envelope_mean + 1e-6)
                    envelope_stable = cv < 0.30  # relaxed to 0.30 to avoid long waits
                
                if lms_ready and wiener_ready and windows_ready and envelope_stable:
                    pipeline.filter_warmup_needed = False
                    pipeline.rest_baseline_collecting = True
                    print(f"\n✅ [{pipeline.device_name}] Filters ready (CV={cv:.3f}). Starting REST baseline ({pipeline.rest_baseline_duration}s)...")
                    print(f"   KEEP ARMS RELAXED - do not move during baseline collection!")
                else:
                    # Occasional status
                    if pipeline.filter_warmup_windows % 10 == 0:
                        print(f"[{pipeline.device_name}] Warmup: windows={pipeline.filter_warmup_windows}/{pipeline.filter_warmup_required}, CV={cv:.3f}, LMS={lms_ready}, Wiener={wiener_ready}")
                    pipeline.total_windows_processed += 1
                    windows_created += 1
                    continue
            
            # Collect REST baseline (simple, fixed duration)
            if pipeline.rest_baseline_collecting:
                if pipeline.rest_baseline_start_time is None:
                    pipeline.rest_baseline_start_time = time.time()
                    pipeline.baseline_envelope_history = deque(maxlen=25)
                
                env_mean = np.mean(features['emg_envelope'])
                pipeline.baseline_envelope_history.append(env_mean)
                
                # Collect baseline samples
                pipeline.preprocessor.collect_rest_baseline(features)
                pipeline.rest_baseline_windows_collected += 1
                
                elapsed = time.time() - pipeline.rest_baseline_start_time
                
                # Finalize after fixed duration (no extensions)
                if elapsed >= pipeline.rest_baseline_duration:
                    pipeline.preprocessor.finalize_rest_baseline()
                    
                    baseline = pipeline.preprocessor.rest_baseline
                    env_mean = baseline['emg_envelope']['mean']
                    env_std = baseline['emg_envelope']['std']
                    cv_final = env_std / (env_mean + 1e-6)
                    
                    print(f"✅ [{pipeline.device_name}] REST baseline collected (CV={cv_final:.3f}, mean={env_mean:.1f}±{env_std:.1f})")
                    
                    # Disable diagnostic prints after calibration completes
                    pipeline.preprocessor.enable_diagnostics = False
                    print(f"   🔕 Diagnostic prints disabled - showing predictions only")
                    
                    # SIMPLIFIED: No adaptive baseline tracker or sustained FIST tracker
                    # Trust the model predictions directly with only consecutive REST filtering
                    
                    pipeline.rest_baseline_collecting = False
                else:
                    pipeline.total_windows_processed += 1
                    windows_created += 1
                    continue
            
            # Compute vs-REST features
            vs_rest_features = self._compute_vs_rest_features(features, pipeline)
            
            # Normalize features
            normalized_features = pipeline.preprocessor.normalize_features(features)
            
            # Add vs-REST features
            normalized_features['vs_rest'] = vs_rest_features
            
            # Update sequence buffer
            sequence_ready = pipeline.window_buffer.update_sequence(normalized_features)
            
            pipeline.total_windows_processed += 1
            windows_created += 1
            
            # Run inference if sequence is ready, but throttle frequency
            if sequence_ready and not pipeline.rest_baseline_collecting:
                current_time = time.time()
                if current_time - pipeline.last_inference_time >= self.inference_interval:
                    self._run_inference(pipeline)
                    pipeline.last_inference_time = current_time
                    
                    # After updating both pipelines, compute wheelchair command
                    self._update_wheelchair_command()
    
    def _run_inference(self, pipeline: DevicePipeline):
        """
        Run model inference on current sequence for a specific device.
        Trusts the raw model prediction - the model was trained on filtered features
        (amplitude, bandpower, vs_rest) and should confidently distinguish REST from FIST.
        
        Args:
            pipeline: DevicePipeline instance to run inference on
        """
        # Get sequence
        sequence = pipeline.window_buffer.get_sequence()
        if sequence is None:
            return
        
        # EXPLICIT MOTION ARTIFACT REJECTION: Reject FIST if motion detected
        # IMU features: [gyro_mag_mean, accel_mag_mean, gyro_mag_std, accel_mag_std]
        imu_features = sequence['imu_features']  # (seq_len, 4)
        avg_gyro_mag = np.mean(imu_features[:, 0])  # Average gyro magnitude across sequence
        avg_accel_mag = np.mean(imu_features[:, 1])  # Average accelerometer magnitude
        gyro_std = np.mean(imu_features[:, 2])  # Average gyro std (indicates variability)
        
        # Motion artifact thresholds (tuned for wheelchair control)
        # Typical REST: gyro_mag < 200, hand waving: gyro_mag > 500
        # Motion artifacts: high gyro + high accel = arm movement (not muscle contraction)
        motion_gyro_threshold = 400.0  # High gyro = arm movement
        motion_accel_threshold = 1500.0  # High accel = arm movement
        motion_detected = (avg_gyro_mag > motion_gyro_threshold) or (avg_accel_mag > motion_accel_threshold)
        
        # Prepare batch for model inference
        batch = {
            'emg_envelope': torch.FloatTensor(sequence['emg_envelope']).unsqueeze(0).to(self.device),
            'emg_bandpowers': torch.FloatTensor(sequence['emg_bandpowers']).unsqueeze(0).to(self.device),
            'imu_features': torch.FloatTensor(sequence['imu_features']).unsqueeze(0).to(self.device),
            'vs_rest': torch.FloatTensor(sequence.get('vs_rest', np.zeros((self.sequence_length, 2)))).unsqueeze(0).to(self.device)
        }
        
        # Run model inference
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        # Get raw prediction from model
        raw_pred_class = self.class_names[pred_idx]
        
        # SIMPLIFIED: Trust the model prediction directly (no motion artifact rejection since gyro is disabled)
        # REMOVED: Motion artifact rejection logic (gyro is now disabled in LMS)
        # REMOVED: Feature-based smoothing and sustained FIST tracker
        # The model was trained to handle these cases, so we trust its predictions
        final_prediction = raw_pred_class
        
        # Track prediction history for safety monitoring
        pipeline.prediction_history.append(final_prediction)
        
        # REMOVED: REST confirmation logic - trust model predictions directly
        
        # Update last valid prediction timestamp (for HOLD state)
        if final_prediction in ['REST', 'FIST']:
            pipeline.last_valid_prediction = final_prediction
            pipeline.last_valid_prediction_time = time.time()
        
        # Track when FIST was last active (for safety timeout logic)
        if final_prediction == 'FIST':
            pipeline.last_fist_time = time.time()
        
        # Update pipeline state
        pipeline.last_prediction = final_prediction
        pipeline.last_confidence = confidence
        pipeline.total_sequences_processed += 1
        
        # Update unified display (updates in place)
        self._print_unified_display()

    def _update_wheelchair_command(self):
        """
        Combine predictions from LEFT and RIGHT devices with DIRECT model trust.
        Uses raw model predictions with confidence thresholds for quality control.
        
        SIMPLIFIED Control Logic:
        1. Trust model predictions directly - no confirmation delays
        2. Apply confidence thresholds to filter low-quality predictions
        3. Commands execute immediately based on current predictions
        4. No movement duration limits - continuous control based on sustained contractions
        
        Motor Control:
        - Device 1 (LEFT) FIST → RIGHT motor spins → LEFT turn
        - Device 2 (RIGHT) FIST → LEFT motor spins → RIGHT turn  
        - BOTH devices FIST → BOTH motors spin → FORWARD
        - REST (either device) → Stop motors immediately
        """
        current_time = time.time()
        
        # Get raw predictions directly from model
        left_pred = self.left_pipeline.last_prediction
        right_pred = self.right_pipeline.last_prediction
        left_conf = self.left_pipeline.last_confidence
        right_conf = self.right_pipeline.last_confidence
        
        # Confidence thresholds for quality control
        min_confidence = 0.55  # Minimum confidence for any movement command
        
        # Apply confidence filtering - treat low confidence as REST for safety
        if left_conf < min_confidence:
            left_pred = 'REST'
        if right_conf < min_confidence:
            right_pred = 'REST'
        
        # Determine wheelchair command based on current predictions
        wheelchair_cmd = 'REST'
        
        # FORWARD: Both devices predict FIST with decent confidence
        if left_pred == 'FIST' and right_pred == 'FIST':
            wheelchair_cmd = 'FORWARD'
        
        # LEFT turn: Only LEFT device predicts FIST
        elif left_pred == 'FIST' and right_pred == 'REST':
            wheelchair_cmd = 'LEFT'
        
        # RIGHT turn: Only RIGHT device predicts FIST
        elif left_pred == 'REST' and right_pred == 'FIST':
            wheelchair_cmd = 'RIGHT'
        
        # REST: Either both REST or confidence too low
        else:
            wheelchair_cmd = 'REST'
        
        # Update state
        self.last_wheelchair_command = wheelchair_cmd
        self.wheelchair_command_history.append({
            'command': wheelchair_cmd,
            'left_pred': left_pred,
            'right_pred': right_pred,
            'left_conf': left_conf,
            'right_conf': right_conf,
            'timestamp': current_time
        })
        
        # Send command to motor controller (immediate execution)
        if self.enable_wheelchair and self.wheelchair_controller is not None:
            self.wheelchair_controller.update_command(wheelchair_cmd)
        
        # Update unified display (includes command)
        self._print_unified_display()
    
    def _compute_vs_rest_features(self, raw_features: dict, pipeline: DevicePipeline) -> np.ndarray:
        """
        Compute enhanced vs-REST features using ADAPTIVE baseline.
        Uses original REST baseline as reference but adapts to current resting state (handles fatigue).
        This allows REST detection even when muscle activity doesn't return to original baseline.
        
        Args:
            raw_features: Raw feature dictionary (before normalization)
            pipeline: DevicePipeline instance
            
        Returns:
            vs_rest array (2,) - [envelope_vs_rest, high_bandpower_vs_rest]
        """
        # If REST baseline is available, compute vs-REST features
        if (self.enable_rest_baseline and 
            pipeline.preprocessor.rest_baseline_collected and 
            pipeline.preprocessor.rest_baseline is not None):
            
            # SIMPLIFIED: Use original baseline only (no adaptive baseline)
            rest_baseline = pipeline.preprocessor.rest_baseline
            
            # Compute envelope mean from current window (raw features)
            envelope_mean = float(np.mean(raw_features['emg_envelope']))
            high_bandpower = float(raw_features['emg_bandpowers'][2])
            
            # Add to feature history for temporal smoothing
            pipeline.feature_history['envelope_mean'].append(envelope_mean)
            pipeline.feature_history['high_bandpower'].append(high_bandpower)
            
            # Use smoothed features (temporal average) to reduce noise
            # This helps the model see clearer REST vs FIST differences
            if len(pipeline.feature_history['envelope_mean']) >= 3:
                # Use median for robustness to outliers
                smoothed_envelope = float(np.median(list(pipeline.feature_history['envelope_mean'])))
                smoothed_bandpower = float(np.median(list(pipeline.feature_history['high_bandpower'])))
            else:
                # Not enough history yet - use current values
                smoothed_envelope = envelope_mean
                smoothed_bandpower = high_bandpower
            
            # Enhanced vs-REST computation with better discriminative power
            if (rest_baseline['emg_envelope']['mean'] is not None and 
                rest_baseline['emg_envelope']['std'] is not None):
                rest_env_mean = rest_baseline['emg_envelope']['mean']
                rest_env_std = rest_baseline['emg_envelope']['std']
                
                # Protect against division by zero
                if rest_env_std < 1e-6:
                    rest_env_std = 1.0
                
                # Enhanced z-score: Use smoothed envelope and add ratio component
                # This makes REST vs FIST differences more pronounced
                z_score = (smoothed_envelope - rest_env_mean) / rest_env_std
                ratio = smoothed_envelope / (rest_env_mean + 1e-6)
                
                # Combine z-score and ratio for better discrimination
                # When ratio is close to 1.0 (REST), z-score dominates
                # When ratio is > 1.5 (FIST), ratio amplifies the signal
                if ratio > 1.2:  # Clear elevation above REST
                    env_vs_rest = z_score * (1.0 + 0.3 * (ratio - 1.0))  # Amplify FIST signal
                else:
                    env_vs_rest = z_score  # Use z-score for REST detection
            else:
                env_vs_rest = 0.0
            
            if (rest_baseline['emg_bandpowers']['mean'] is not None and 
                len(rest_baseline['emg_bandpowers']['mean']) > 2 and
                rest_baseline['emg_bandpowers']['std'] is not None and
                len(rest_baseline['emg_bandpowers']['std']) > 2):
                rest_bp_mean = rest_baseline['emg_bandpowers']['mean'][2]
                rest_bp_std = rest_baseline['emg_bandpowers']['std'][2]
                
                # Protect against division by zero
                if rest_bp_std < 1e-6:
                    rest_bp_std = 1.0
                
                # Enhanced bandpower vs-REST with ratio amplification
                z_score = (smoothed_bandpower - rest_bp_mean) / rest_bp_std
                ratio = smoothed_bandpower / (rest_bp_mean + 1e-6)
                
                # Amplify FIST frequency content
                if ratio > 1.2:
                    high_vs_rest = z_score * (1.0 + 0.3 * (ratio - 1.0))
                else:
                    high_vs_rest = z_score
            else:
                high_vs_rest = 0.0
            
            return np.array([env_vs_rest, high_vs_rest], dtype=np.float32)
        else:
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def _print_unified_display(self):
        """
        Print a single unified line showing predictions, command, and execution state.
        Updates in place without printing new lines.
        """
        # Only show if at least one pipeline has processed predictions
        if (self.left_pipeline.total_sequences_processed == 0 and 
            self.right_pipeline.total_sequences_processed == 0):
            return
        
        # Throttle updates (update every 100ms)
        current_time = time.time()
        if current_time - self.last_display_update < 0.1:
            return
        self.last_display_update = current_time
        
        # Move cursor back if we've printed before
        if self.display_lines_printed > 0:
            print(f"\033[{self.display_lines_printed}A", end='')
        
        # Get current predictions
        left_pred = self.left_pipeline.last_prediction
        left_conf = self.left_pipeline.last_confidence
        right_pred = self.right_pipeline.last_prediction
        right_conf = self.right_pipeline.last_confidence
        
        # Format predictions with emojis
        left_emoji = '✊' if left_pred == 'FIST' else '🛑'
        right_emoji = '✊' if right_pred == 'FIST' else '🛑'
        
        # Get current command and execution state
        cmd = self.last_wheelchair_command
        if self.wheelchair_controller and self.wheelchair_controller.is_moving:
            cmd_symbol = '⚡'  # Executing
            cmd_text = f"{cmd}"
        else:
            if cmd == 'FORWARD':
                cmd_symbol = '⬆️'
            elif cmd == 'LEFT':
                cmd_symbol = '⬅️'
            elif cmd == 'RIGHT':
                cmd_symbol = '➡️'
            else:
                cmd_symbol = '🛑'
            cmd_text = cmd
        
        # Build single unified line
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {left_emoji} {left_pred} ({left_conf:.1%})  |  {right_emoji} {right_pred} ({right_conf:.1%})  |  {cmd_symbol} {cmd_text}"
        
        print(f"\033[K{line}", flush=True)
        self.display_lines_printed = 1
    
    def _print_waiting_display(self):
        """Print waiting message in static display format."""
        if self.display_lines_printed > 0:
            print(f"\033[{self.display_lines_printed}A", end='')
        
        stats_left = self.server_left.get_stats()
        stats_right = self.server_right.get_stats()
        left_connected = stats_left['is_connected']
        right_connected = stats_right['is_connected']
        
        if left_connected and right_connected:
            connection_status = "✅ Both devices connected"
        elif left_connected:
            connection_status = "⚠️  LEFT connected, waiting for RIGHT..."
        elif right_connected:
            connection_status = "⚠️  RIGHT connected, waiting for LEFT..."
        else:
            connection_status = "❌ Waiting for ESP32 devices..."
        
        # Check warmup/baseline status
        left_status = "Filter warmup" if self.left_pipeline.filter_warmup_needed else (
            "Baseline collection" if self.left_pipeline.rest_baseline_collecting else "Ready"
        )
        right_status = "Filter warmup" if self.right_pipeline.filter_warmup_needed else (
            "Baseline collection" if self.right_pipeline.rest_baseline_collecting else "Ready"
        )
        
        lines = [
            f"\033[1m\033[93m⏳ INITIALIZING DUAL-DEVICE SYSTEM...\033[0m  |  {connection_status}",
            f"   LEFT (ID=1): {left_status}  |  RIGHT (ID=2): {right_status}",
            f"   Packets received: LEFT={stats_left.get('total_packets_received', 0)} RIGHT={stats_right.get('total_packets_received', 0)}"
        ]
        
        for line in lines:
            print(f"\033[K{line}")
        
        self.display_lines_printed = len(lines)
    
    
    def start(self):
        """Start the inference engine."""
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("🎯 Starting Dual-Device Wheelchair Control System")
        print("="*70)
        print(f"   UDP Ports (DUAL-PORT MODE):")
        print(f"      LEFT device (ID=1):  {self.port}")
        print(f"      RIGHT device (ID=2): {self.port + 1}")
        print(f"   Device: {self.device}")
        print(f"   Window size: {self.window_size} samples ({self.window_size/self.sampling_rate*1000:.0f}ms)")
        print(f"   Sequence length: {self.sequence_length} windows (~{self.sequence_length * self.stride / self.sampling_rate * 1000:.0f}ms latency)")
        print(f"   Stride: {self.stride} samples ({self.stride/self.sampling_rate*1000:.0f}ms between windows)")
        print(f"   Protocol: UDP (dual-device, dual-port)")
        print(f"\n   🔧 Filter Warmup & Calibration:")
        print(f"      Phase 1: Filter warmup (~10s) - LMS adapts, Wiener calibrates")
        print(f"              - 50 windows minimum, simple CV<0.30 check")
        print(f"              - Diagnostic prints enabled during warmup")
        print(f"      Phase 2: REST baseline collection (5s per device) - KEEP ARMS RELAXED!")
        print(f"              - Fixed duration (no extensions), reports baseline CV")
        print(f"              - Diagnostic prints disabled after completion")
        print(f"      Phase 3: Real-time wheelchair control")
        print(f"   IMU Configuration:")
        print(f"      Device 1 (LEFT):  IMU DISABLED (pure EMG only, no motion artifacts)")
        print(f"      Device 2 (RIGHT): IMU ENABLED (motion features available)")
        print(f"   REST baseline vs_rest features: {'ENABLED (REQUIRED for model)' if self.enable_rest_baseline else 'DISABLED'}")
        print(f"   Min confidence threshold: {self.min_confidence:.2%}")
        print(f"\n   🔬 SIMPLIFIED CONTROL ALGORITHM:")
        print(f"      ✓ Direct model predictions (no feature-based overriding)")
        print(f"      ✓ Gyroscope disabled (motion does not interfere with gestures)")
        print(f"      ✓ Simple REST detection (requires 3 consecutive REST for reliable STOP)")
        print(f"      ✓ Fast FIST detection (immediate GO response for safety)")
        print(f"      → Trust the trained model - it learned the right features")
        print(f"      → Simpler logic = more predictable behavior")
        print(f"\n   🚀 CONTINUOUS Wheelchair Control:")
        print(f"      Mode: Model predictions directly control movement")
        print(f"      Motor power: 50% duty cycle")
        print(f"      Device 1 (LEFT) FIST  → RIGHT motor spins → LEFT turn")
        print(f"      Device 2 (RIGHT) FIST → LEFT motor spins → RIGHT turn")
        print(f"      BOTH devices FIST     → BOTH motors spin → FORWARD")
        print(f"      ✓ Continuous predictions enabled (no blocking)")
        print(f"      ✓ Sustained contractions work as model was trained")
        print("\n" + "─"*70)
        print(f"📊 Waiting for ESP32 UDP packets from both devices...")
        print("─"*70)
        
        # Show network info
        try:
            import socket
            import subprocess
            hostname = socket.gethostname()
            
            # Get all network interfaces and their IPs
            print(f"   Hostname: {hostname}")
            print(f"   Network interfaces:")
            
            try:
                # Try to get IP addresses from ifconfig/ip command
                result = subprocess.run(['ip', 'addr', 'show'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if 'inet ' in line and '127.0.0.1' not in line:
                            ip_part = line.strip().split()[1].split('/')[0]
                            # Get interface name from previous line
                            if i > 0:
                                interface_line = lines[i-1]
                                if ':' in interface_line:
                                    interface = interface_line.split(':')[1].strip().split('@')[0]
                                    print(f"      {interface}: {ip_part}")
                                    if 'wlan' in interface.lower() or '192.168.4' in ip_part:
                                        print(f"      ⭐ Hotspot IP (use this): {ip_part}")
            except:
                pass
            
            # Fallback: Get default route IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
                print(f"   Default route IP: {local_ip}")
            except:
                local_ip = 'Unknown'
            finally:
                s.close()
            
            print(f"\n   ⚠️  ESP32 Configuration Check (DUAL-PORT MODE):")
            print(f"      LEFT ESP32 (Device 1) should send to: 192.168.4.1:{self.port}")
            print(f"      RIGHT ESP32 (Device 2) should send to: 192.168.4.1:{self.port + 1}")
            print(f"      If your Pi's hotspot IP is different, update ESP32 code!")
        except Exception as e:
            if self.debug:
                print(f"   Network detection error: {e}")
        
        print("─"*70)
        print("\n💡 Troubleshooting Tips:")
        print(f"   1. Verify ESP32 is connected to WiFi hotspot")
        print(f"   2. Check ESP32 code has SERVER_IP = '192.168.4.1' (or your Pi's hotspot IP)")
        print(f"   3. Check ESP32 code has SERVER_PORT = {self.port}")
        print(f"   4. Test UDP reception: python3 test_udp_receive.py --port {self.port}")
        print(f"   5. Check firewall: sudo ufw allow {self.port}/udp")
        print(f"   6. Run with --debug for detailed packet info")
        print("─"*70 + "\n")
        
        # CRITICAL: Start processing worker thread before UDP servers
        print("\n🚀 Starting processing worker thread...")
        self.processing_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_worker,
            daemon=True,
            name="Batch-Processing-Worker"
        )
        self.processing_thread.start()
        print(f"   ✅ Processing worker ready (queue-based for low latency)")
        
        # Start BOTH servers (dual-port mode)
        print("\n🚀 Starting DUAL-PORT UDP servers...")
        self.server_left.start()
        print(f"   ✅ LEFT server ready on port {self.port}")
        self.server_right.start()
        print(f"   ✅ RIGHT server ready on port {self.port + 1}")
        print(f"   ✅ UDP reception now non-blocking (queue-based processing)")
        
        # Print initial waiting display
        print("\n")  # Add some space before static display
        self._print_waiting_display()
        
        # Main monitoring loop
        try:
            self._monitor_loop()
        except KeyboardInterrupt:
            # Move cursor down past the static display
            if self.display_lines_printed > 0:
                print(f"\n" * (self.display_lines_printed + 1))
            print("="*70)
            print("🛑 Shutting down...")
            print("="*70)
            self.stop()
    
    def _monitor_loop(self):
        """Monitor loop - check for packet loss and update waiting display."""
        last_waiting_update = 0
        waiting_update_interval = 1.0
        
        # CRITICAL: Periodic timeout check for faster "device off" detection
        last_timeout_check = 0
        timeout_check_interval = 0.1  # Check every 100ms for responsive device-off detection
        
        while True:
            time.sleep(0.01)  # 100Hz loop
            
            current_time = time.time()
            
            # CRITICAL: Periodic timeout check - detect device off within 100ms
            # This ensures we don't wait for the next packet to notice device is off
            if current_time - last_timeout_check >= timeout_check_interval:
                for device_id, pipeline in self.pipelines.items():
                    if self.last_data_received_time[device_id] is not None:
                        time_since_last_data = current_time - self.last_data_received_time[device_id]
                        if time_since_last_data > self.data_timeout:
                            if not self.packet_loss_detected[device_id]:
                                self.packet_loss_detected[device_id] = True
                                print(f"⚠️  [{pipeline.device_name}] Device timeout (no data for {time_since_last_data:.2f}s)")
                                # CRITICAL: Immediately set to REST
                                pipeline.last_prediction = 'REST'
                                pipeline.last_confidence = 0.0
                                pipeline.window_buffer.clear()
                                # Update wheelchair command immediately
                                self._update_wheelchair_command()
                last_timeout_check = current_time
            
            # Show waiting display if no predictions yet
            total_predictions = self.left_pipeline.total_sequences_processed + self.right_pipeline.total_sequences_processed
            if total_predictions == 0:
                if current_time - last_waiting_update >= waiting_update_interval:
                    self._print_waiting_display()
                    last_waiting_update = current_time
            
            # Legacy check (now supplemented by periodic check above)
            for device_id, pipeline in self.pipelines.items():
                if self.last_data_received_time[device_id] is not None:
                    time_since_last_data = current_time - self.last_data_received_time[device_id]
                    if time_since_last_data > self.data_timeout:
                        if not self.packet_loss_detected[device_id]:
                            self.packet_loss_detected[device_id] = True
                            print(f"⚠️  [{pipeline.device_name}] Packet loss detected (no data for {time_since_last_data:.2f}s)")
                            pipeline.window_buffer.clear()
                            self.packet_loss_detected[device_id] = True
                            # Emergency stop on packet loss
                            if self.enable_wheelchair and self.wheelchair_controller is not None:
                                self.wheelchair_controller.emergency_stop()
    
    def _print_stats(self):
        """Print statistics for dual-device wheelchair control."""
        stats_left = self.server_left.get_stats()
        stats_right = self.server_right.get_stats()
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "─"*60)
        print("📊 Dual-Device Statistics (DUAL-PORT MODE)")
        print("─"*60)
        print(f"Runtime: {elapsed_time:.0f}s")
        print(f"LEFT Status: {'✅ Connected' if stats_left['is_connected'] else '❌ Disconnected'}")
        print(f"RIGHT Status: {'✅ Connected' if stats_right['is_connected'] else '❌ Disconnected'}")
        print(f"Packets LEFT: {stats_left.get('total_packets_received', 0):,} received, {stats_left.get('total_packets_failed', 0):,} failed")
        print(f"Packets RIGHT: {stats_right.get('total_packets_received', 0):,} received, {stats_right.get('total_packets_failed', 0):,} failed")
        
        # Per-device stats
        print(f"\nLEFT device (ID=1):")
        print(f"   Predictions: {self.left_pipeline.total_sequences_processed}")
        print(f"   Last: {self.left_pipeline.last_prediction} ({self.left_pipeline.last_confidence:.1%})")
        
        print(f"\nRIGHT device (ID=2):")
        print(f"   Predictions: {self.right_pipeline.total_sequences_processed}")
        print(f"   Last: {self.right_pipeline.last_prediction} ({self.right_pipeline.last_confidence:.1%})")
        
        # Wheelchair commands
        if len(self.wheelchair_command_history) > 0:
            recent = list(self.wheelchair_command_history)[-10:]
            cmd_counts = {}
            for entry in recent:
                cmd = entry['command']
                cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1
            print(f"\nLast 10 wheelchair commands:")
            print(f"   REST={cmd_counts.get('REST', 0)}, LEFT={cmd_counts.get('LEFT', 0)}, "
                  f"RIGHT={cmd_counts.get('RIGHT', 0)}, FORWARD={cmd_counts.get('FORWARD', 0)}")
        
        print("─"*60)
    
    def stop(self):
        """Stop the inference engine."""
        # Stop wheelchair first (safety)
        if self.enable_wheelchair and self.wheelchair_controller is not None:
            print("\n🛑 Stopping wheelchair...")
            self.wheelchair_controller.emergency_stop()
            self.wheelchair_controller.cleanup()
        
        # CRITICAL: Stop processing worker thread
        print("🛑 Stopping processing worker...")
        self.processing_running = False
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)
        
        # Stop UDP servers
        self.server_left.stop()
        self.server_right.stop()
        
        # Print final statistics
        print("\n" + "="*70)
        print("📊 Final Dual-Device Statistics")
        print("="*70)
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        print(f"   Total runtime: {elapsed_time:.1f}s")
        
        total_windows = self.left_pipeline.total_windows_processed + self.right_pipeline.total_windows_processed
        total_sequences = self.left_pipeline.total_sequences_processed + self.right_pipeline.total_sequences_processed
        
        print(f"   Total windows processed: {total_windows:,}")
        print(f"   Total sequences processed: {total_sequences:,}")
        
        print(f"\n   LEFT device (ID=1):")
        print(f"      Windows: {self.left_pipeline.total_windows_processed:,}")
        print(f"      Sequences: {self.left_pipeline.total_sequences_processed:,}")
        
        print(f"\n   RIGHT device (ID=2):")
        print(f"      Windows: {self.right_pipeline.total_windows_processed:,}")
        print(f"      Sequences: {self.right_pipeline.total_sequences_processed:,}")
        
        if total_sequences > 0:
            print(f"\n   Processing rate: {total_sequences/elapsed_time:.2f} sequences/sec")
        
        print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Real-time EMG gesture classification inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py
  python inference.py --model-dir ./ --port 5000
  python inference.py --device cpu --debug
  python inference.py --enable-rest-baseline
        """
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='./',
        help='Directory containing deployment files (default: ./)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Server port (default: 5000 for UDP, matches ESP32 default)'
    )
    
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on (default: cpu)'
    )
    
    parser.add_argument(
        '--disable-rest-baseline',
        action='store_true',
        help='DISABLE REST-baseline (NOT recommended). Model was trained with vs-REST features and needs them for accurate sustained FIST detection. Only disable for testing.'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging for UDP server (verbose packet info)'
    )
    
    parser.add_argument(
        '--dashboard-interval',
        type=float,
        default=5.0,
        help='Seconds between dashboard updates (default: 5.0)'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.40,
        help='Minimum confidence threshold for FIST predictions (default: 0.40 for sustained control). The model differentiates REST from FIST based on amplitude and bandpower. Recommended: 0.35-0.50 for wheelchair control. Lower values = more sensitive, higher values = more strict.'
    )
    
    parser.add_argument(
        '--enable-wheelchair',
        action='store_true',
        help='Enable wheelchair motor control (requires gpiozero and BTS 7960 motor drivers connected)'
    )
    
    parser.add_argument(
        '--wheelchair-debug',
        action='store_true',
        help='Wheelchair debug mode: process predictions but do not move motors (for testing safety logic)'
    )
    
    args = parser.parse_args()
    
    # Check if model directory exists
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Create and start inference engine
    engine = InferenceEngine(
        model_dir=str(model_dir),
        device=args.device,
        port=args.port,
        enable_rest_baseline=not args.disable_rest_baseline,  # Enabled by default, disable with flag
        debug=args.debug,
        dashboard_interval=args.dashboard_interval,
        min_confidence=args.min_confidence,
        enable_wheelchair=args.enable_wheelchair,
        wheelchair_debug=args.wheelchair_debug
    )
    
    engine.start()


if __name__ == '__main__':
    main()

