"""
UDP server to receive data batches from ESP32 device.
Simpler than TCP - no connection management needed.
Handles JSON batch parsing and data extraction.
ESP32 sends: [{"m":...,"ax":...,"ay":...,"az":...,"gx":...,"gy":...,"gz":...},...]
Also supports CSV format for backward compatibility.
"""

import socket
import json
import numpy as np
from typing import Dict, Optional, Callable
import threading
import time


class UDPServer:
    """
    UDP server for receiving EMG/IMU data from ESP32.
    Receives JSON batches via UDP packets.
    Format: [{"m":muscle_value,"ax":...,"ay":...,"az":...,"gx":...,"gy":...,"gz":...},...]
    Also supports CSV format for backward compatibility.
    """
    
    def __init__(self,
                 host: str = '0.0.0.0',
                 port: int = 5000,
                 batch_callback: Optional[Callable] = None,
                 debug: bool = False):
        """
        Args:
            host: Server host address
            port: Server port
            batch_callback: Callback function for processed batches
                           Signature: callback(device_id, emg_samples, imu_samples)
                           device_id: 1=LEFT ARM, 2=RIGHT ARM
            debug: Enable debug logging
        """
        self.host = host
        self.port = port
        self.batch_callback = batch_callback
        self.debug = debug
        self.socket = None
        self.running = False
        
        # Statistics
        self.total_batches = 0
        self.total_samples = 0
        self.last_batch_time = None
        self.last_client_address = None
        self.total_packets_received = 0
        self.total_packets_failed = 0
        self.first_packet_received = False
        
        # Buffer for fragmented JSON packets (keyed by client address)
        self.packet_buffers: Dict[tuple, str] = {}
        self.buffer_timestamps: Dict[tuple, float] = {}
        self.buffer_timeout = 0.5  # Clear buffer if no packets for 0.5 seconds
        
    def start(self):
        """Start the UDP server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Enable broadcast reception
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(1.0)  # Timeout for periodic checks
            self.running = True
            
            # Get actual bound address
            actual_host, actual_port = self.socket.getsockname()
            print(f"📡 UDP Server started")
            print(f"   Listening on: {actual_host}:{actual_port}")
            print(f"   ESP32 should send to: {actual_host}:{actual_port}")
            if self.debug:
                print(f"   Debug mode: ENABLED")
            print(f"   ✅ Server is ready and waiting for UDP packets...\n")
            
            # Start receiving data in a separate thread
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True, name="UDP-Receive")
            self.receive_thread.start()
        except Exception as e:
            print(f"❌ Failed to start UDP server: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _receive_loop(self):
        """Main UDP receive loop."""
        while self.running:
            try:
                # Receive UDP packet (max 65507 bytes)
                data, client_address = self.socket.recvfrom(65507)
                
                # Update statistics
                self.total_packets_received += 1
                
                # Print first packet notification
                if not self.first_packet_received:
                    self.first_packet_received = True
                    print(f"\n✅ FIRST UDP PACKET RECEIVED!")
                    print(f"   From: {client_address[0]}:{client_address[1]}")
                    print(f"   Size: {len(data)} bytes\n")
                
                if self.debug:
                    print(f"📦 Received UDP packet from {client_address[0]}:{client_address[1]} "
                          f"(size: {len(data)} bytes)")
                
                # Update client info
                self.last_client_address = client_address
                self.last_batch_time = time.time()
                
                # Clean up stale buffers
                current_time = time.time()
                stale_clients = [addr for addr, ts in self.buffer_timestamps.items() 
                                if current_time - ts > self.buffer_timeout]
                for addr in stale_clients:
                    if self.debug:
                        print(f"🧹 Clearing stale buffer for {addr[0]}:{addr[1]}")
                    del self.packet_buffers[addr]
                    del self.buffer_timestamps[addr]
                
                # Decode data
                try:
                    packet_str = data.decode('utf-8')
                    if self.debug:
                        print(f"   First 200 chars: {packet_str[:200]}")
                except UnicodeDecodeError:
                    # Silently handle non-UTF-8 data (likely binary/motion artifacts)
                    self.total_packets_failed += 1
                    if self.debug:
                        print(f"⚠️  Received non-UTF-8 data from {client_address} (size: {len(data)} bytes)")
                        print(f"   Raw bytes (hex): {data[:50].hex()}")
                    continue
                
                # Accumulate packet into buffer for this client
                if client_address not in self.packet_buffers:
                    self.packet_buffers[client_address] = ""
                
                self.packet_buffers[client_address] += packet_str
                self.buffer_timestamps[client_address] = current_time
                
                # Try to parse complete JSON from buffer
                buffer_content = self.packet_buffers[client_address]
                batch_data = []
                
                try:
                    # Try parsing the accumulated buffer
                    json_data = json.loads(buffer_content.strip())
                    
                    if not isinstance(json_data, list):
                        if self.debug:
                            print(f"⚠️  Expected JSON array, got: {type(json_data)}")
                        # Clear buffer since this is not fragmented JSON
                        del self.packet_buffers[client_address]
                        del self.buffer_timestamps[client_address]
                        self.total_packets_failed += 1
                        continue
                    
                    if self.debug:
                        print(f"   ✅ Successfully parsed JSON array with {len(json_data)} samples")
                        print(f"   Buffer size: {len(buffer_content)} chars")
                    
                    # Clear buffer since we successfully parsed
                    del self.packet_buffers[client_address]
                    del self.buffer_timestamps[client_address]
                    
                    # Extract samples from JSON
                    for item in json_data:
                        if not isinstance(item, dict):
                            if self.debug:
                                print(f"⚠️  Expected dict in array, got: {type(item)}")
                            continue
                        
                        try:
                            # Map JSON keys to our format
                            # ESP32 uses "m" for muscle and "id" for device ID
                            sample = {
                                'device_id': int(item.get('id', 0)),  # Device ID (1=LEFT, 2=RIGHT)
                                'muscle': int(item.get('m', item.get('muscle', 0))),
                                'ax': int(item.get('ax', 0)),
                                'ay': int(item.get('ay', 0)),
                                'az': int(item.get('az', 0)),
                                'gx': int(item.get('gx', 0)),
                                'gy': int(item.get('gy', 0)),
                                'gz': int(item.get('gz', 0))
                            }
                            batch_data.append(sample)
                        except (ValueError, TypeError) as e:
                            if self.debug or self.total_packets_failed < 5:
                                print(f"⚠️  Error parsing JSON sample: {item} (error: {e})")
                            self.total_packets_failed += 1
                            continue
                
                except json.JSONDecodeError as e:
                    # JSON is incomplete (fragmented across packets)
                    # Check if buffer looks like JSON (starts with [)
                    if buffer_content.strip().startswith('['):
                        # This is clearly JSON, just fragmented - wait for more packets
                        if self.debug:
                            print(f"   ⏳ JSON incomplete (fragmented), buffering... (buffer size: {len(buffer_content)} chars)")
                        
                        # Check if buffer is getting too large (might indicate a problem)
                        if len(buffer_content) > 100000:  # 100KB limit
                            if self.debug or self.total_packets_failed < 5:
                                print(f"⚠️  Buffer too large ({len(buffer_content)} chars), clearing...")
                            del self.packet_buffers[client_address]
                            del self.buffer_timestamps[client_address]
                            self.total_packets_failed += 1
                        
                        # Wait for more packets to complete JSON
                        continue
                    
                    # Buffer doesn't start with [ - might be CSV format
                    # Only try CSV if buffer is small and doesn't look like JSON
                    if len(buffer_content) < 1000:
                        # Clear buffer and try CSV on just this packet
                        del self.packet_buffers[client_address]
                        del self.buffer_timestamps[client_address]
                        
                        if self.debug:
                            print(f"⚠️  Small buffer, trying CSV format as fallback...")
                        
                        # Parse CSV format: each line is a sample
                        # Format: muscle,ax,ay,az,gx,gy,gz\n
                        lines = packet_str.strip().split('\n')
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Skip if line looks like JSON (starts with [ or {)
                            if line.startswith('[') or line.startswith('{'):
                                continue
                            
                            # Parse CSV line: muscle,ax,ay,az,gx,gy,gz
                            parts = line.split(',')
                            if len(parts) >= 7:
                                try:
                                    sample = {
                                        'device_id': 0,  # CSV format doesn't include device_id
                                        'muscle': int(parts[0]),
                                        'ax': int(parts[1]),
                                        'ay': int(parts[2]),
                                        'az': int(parts[3]),
                                        'gx': int(parts[4]),
                                        'gy': int(parts[5]),
                                        'gz': int(parts[6])
                                    }
                                    batch_data.append(sample)
                                except (ValueError, IndexError) as e:
                                    # Suppress CSV parsing errors - likely JSON fragment
                                    if self.debug:
                                        print(f"⚠️  Error parsing CSV line: {line[:50]}... (error: {e})")
                                    continue
                            else:
                                if self.debug:
                                    print(f"⚠️  Line has {len(parts)} parts, expected 7: {line[:50]}")
                    else:
                        # Large buffer that's not valid JSON - might be corrupted
                        if self.debug or self.total_packets_failed < 5:
                            print(f"⚠️  Large invalid buffer, clearing...")
                        del self.packet_buffers[client_address]
                        del self.buffer_timestamps[client_address]
                        self.total_packets_failed += 1
                        continue
                    
                    if len(batch_data) == 0:
                        # No valid samples parsed - this is expected for fragmented JSON
                        continue
                
                # Process batch if we have valid samples
                if len(batch_data) > 0:
                    if self.debug:
                        print(f"✅ Processed batch with {len(batch_data)} samples")
                    # Remove always-on print to avoid disrupting display
                    self._process_batch(batch_data, client_address)
                else:
                    # No valid samples, but this might be expected if JSON is fragmented
                    # Don't count as failed if we're still buffering
                    if client_address not in self.packet_buffers:
                        if self.debug:
                            print(f"⚠️  No valid samples in packet")
                        self.total_packets_failed += 1
                
            except socket.timeout:
                # Timeout is expected, continue
                continue
            except socket.error as e:
                if self.running:
                    error_str = str(e).lower()
                    if "bad file descriptor" not in error_str:
                        print(f"⚠️  UDP socket error: {e}")
                        if self.debug:
                            import traceback
                            traceback.print_exc()
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ Error receiving UDP packet: {e}")
                    import traceback
                    traceback.print_exc()
                self.total_packets_failed += 1
                continue
    
    def _process_batch(self, batch_data: list, client_address: tuple):
        """
        Process a batch of samples.
        Groups samples by device_id and calls callback for each device separately.
        
        Args:
            batch_data: List of sample dictionaries
            client_address: Tuple of (ip, port) of sender
        """
        if not isinstance(batch_data, list) or len(batch_data) == 0:
            return
        
        # Group samples by device_id
        device_samples = {}  # {device_id: [(emg, imu), ...]}
        
        for sample in batch_data:
            device_id = sample.get('device_id', 0)
            
            # Extract EMG
            if 'muscle' in sample:
                emg_value = float(sample['muscle'])
            else:
                emg_value = 0.0
            
            # Extract IMU (ax, ay, az, gx, gy, gz)
            imu = [
                float(sample.get('ax', 0)),
                float(sample.get('ay', 0)),
                float(sample.get('az', 0)),
                float(sample.get('gx', 0)),
                float(sample.get('gy', 0)),
                float(sample.get('gz', 0))
            ]
            
            # Group by device
            if device_id not in device_samples:
                device_samples[device_id] = {'emg': [], 'imu': []}
            
            device_samples[device_id]['emg'].append(emg_value)
            device_samples[device_id]['imu'].append(imu)
        
        # Update statistics
        self.total_batches += 1
        self.total_samples += len(batch_data)
        self.last_batch_time = time.time()
        
        # Call callback for each device separately
        if self.batch_callback is not None:
            for device_id, samples in device_samples.items():
                emg_array = np.array(samples['emg'], dtype=np.float32)
                imu_array = np.array(samples['imu'], dtype=np.float32)
                self.batch_callback(device_id, emg_array, imu_array)
    
    def is_connected(self) -> bool:
        """Check if we're receiving data (UDP is connectionless, so check recent activity)."""
        if self.last_batch_time is None:
            return False
        
        # Consider connected if we received data in last 5 seconds
        return (time.time() - self.last_batch_time) < 5.0
    
    def get_stats(self) -> Dict:
        """Get server statistics."""
        return {
            'total_batches': self.total_batches,
            'total_samples': self.total_samples,
            'is_connected': self.is_connected(),
            'last_batch_time': self.last_batch_time,
            'last_client': self.last_client_address,
            'total_packets_received': self.total_packets_received,
            'total_packets_failed': self.total_packets_failed
        }
    
    def stop(self):
        """Stop the UDP server."""
        self.running = False
        
        if self.socket is not None:
            try:
                self.socket.close()
            except:
                pass
        
        print("🛑 UDP Server stopped")

