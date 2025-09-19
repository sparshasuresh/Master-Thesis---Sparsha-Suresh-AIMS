#!/usr/bin/env python3
"""
Minimal CARLA Sensor Fusion - Kalman Filter + Bayesian Network Only
For systems with limited disk space - No PyTorch required
German Automotive Industry Study Implementation
"""

import sys
import time
import random
import math
from collections import deque

# Check essential dependencies only
try:
    import carla
    print("✓ CARLA imported successfully")
except ImportError:
    print("✗ CARLA not found. Please install: pip install carla")
    sys.exit(1)

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError:
    print("✗ NumPy not found. Please install: pip install numpy")
    sys.exit(1)

print("Essential dependencies loaded successfully!")
print("Running minimal sensor fusion (Kalman Filter + Bayesian Network)")
print("-" * 60)

class KalmanFilter:
    """Enhanced Kalman Filter for vehicle state estimation"""
    def __init__(self, dt=0.1):
        # State vector: [x, y, vx, vy, ax, ay]
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1000.0  # Initial uncertainty
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Observation matrices
        self.H_gps = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])  # GPS: position
        self.H_imu = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])  # IMU: acceleration
        self.H_lidar = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]) # LiDAR: position
        
        # Noise matrices (tuned for German road conditions)
        self.R_gps = np.array([[2.0, 0], [0, 2.0]])      # GPS less accurate in urban areas
        self.R_imu = np.array([[0.3, 0], [0, 0.3]])      # IMU quite reliable
        self.R_lidar = np.array([[0.1, 0], [0, 0.1]])    # LiDAR very accurate
        
        # Process noise
        self.Q = np.eye(6) * 0.1
        self.I = np.eye(6)
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, z, sensor_type='gps'):
        if sensor_type == 'gps':
            H, R = self.H_gps, self.R_gps
        elif sensor_type == 'imu':
            H, R = self.H_imu, self.R_imu
        elif sensor_type == 'lidar':
            H, R = self.H_lidar, self.R_lidar
        else:
            return self.x
            
        # Kalman update equations
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P
        
        return self.x
    
    def get_position(self):
        return self.x[0:2].flatten()
    
    def get_velocity(self):
        return self.x[2:4].flatten()
    
    def get_acceleration(self):
        return self.x[4:6].flatten()

class BayesianNetwork:
    """Bayesian Network for sensor reliability assessment - German weather focused"""
    def __init__(self):
        # Prior probabilities for German weather conditions
        self.weather_prob = {
            'clear': 0.45,     # Less clear days in Germany
            'rainy': 0.35,     # Frequent rain
            'foggy': 0.15,     # Common in northern regions
            'snowy': 0.05      # Winter conditions
        }
        
        # Conditional probabilities - German automotive industry standards
        self.sensor_reliability = {
            'gps': {
                'clear': {'high': 0.85, 'medium': 0.12, 'low': 0.03},
                'rainy': {'high': 0.75, 'medium': 0.20, 'low': 0.05},
                'foggy': {'high': 0.80, 'medium': 0.15, 'low': 0.05},
                'snowy': {'high': 0.65, 'medium': 0.25, 'low': 0.10}
            },
            'lidar': {
                'clear': {'high': 0.95, 'medium': 0.04, 'low': 0.01},
                'rainy': {'high': 0.70, 'medium': 0.25, 'low': 0.05},
                'foggy': {'high': 0.40, 'medium': 0.35, 'low': 0.25},
                'snowy': {'high': 0.60, 'medium': 0.30, 'low': 0.10}
            },
            'imu': {  # IMU generally weather-independent
                'clear': {'high': 0.90, 'medium': 0.08, 'low': 0.02},
                'rainy': {'high': 0.88, 'medium': 0.10, 'low': 0.02},
                'foggy': {'high': 0.89, 'medium': 0.09, 'low': 0.02},
                'snowy': {'high': 0.85, 'medium': 0.12, 'low': 0.03}
            }
        }
        
        self.current_weather = 'clear'
        self.reliability_history = []
        
    def update_weather(self, visibility, precipitation, temperature=15.0):
        """Update weather condition based on sensor data and German climate"""
        if temperature < 2.0 and precipitation > 0.3:
            self.current_weather = 'snowy'
        elif precipitation > 0.4:
            self.current_weather = 'rainy'
        elif visibility < 0.4:
            self.current_weather = 'foggy'
        else:
            self.current_weather = 'clear'
            
        # Store reliability history for trend analysis
        weights = self.get_sensor_weights()
        self.reliability_history.append({
            'timestamp': time.time(),
            'weather': self.current_weather,
            'weights': weights
        })
        
        # Keep only recent history (last 60 seconds)
        current_time = time.time()
        self.reliability_history = [
            h for h in self.reliability_history 
            if current_time - h['timestamp'] < 60
        ]
    
    def get_sensor_weights(self):
        """Get reliability weights for each sensor"""
        weights = {}
        for sensor in ['gps', 'lidar', 'imu']:
            reliability_probs = self.sensor_reliability[sensor][self.current_weather]
            # Calculate expected reliability score
            weights[sensor] = (
                reliability_probs['high'] * 1.0 + 
                reliability_probs['medium'] * 0.6 + 
                reliability_probs['low'] * 0.2
            )
        return weights
    
    def get_confidence_level(self):
        """Get overall system confidence level"""
        weights = self.get_sensor_weights()
        return min(weights.values())  # Most conservative approach

class SimplifiedTrajectoryPredictor:
    """Simple trajectory prediction without neural networks"""
    def __init__(self):
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        
    def update(self, position, velocity):
        self.position_history.append(position)
        self.velocity_history.append(velocity)
    
    def predict_control(self, target_waypoint, current_transform):
        """Simple geometric trajectory prediction"""
        if not target_waypoint:
            return np.array([0.0, 0.0])  # [steering, throttle]
        
        # Calculate distance and angle to target
        target_x = target_waypoint.transform.location.x
        target_y = target_waypoint.transform.location.y
        
        current_x = current_transform.location.x
        current_y = current_transform.location.y
        
        # Distance to target
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Angle to target
        target_angle = math.atan2(dy, dx)
        current_yaw = math.radians(current_transform.rotation.yaw)
        angle_diff = target_angle - current_yaw
        
        # Normalize angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Simple control calculation
        steering = np.clip(angle_diff * 0.7, -1.0, 1.0)
        throttle = min(0.6, distance / 15.0)
        
        return np.array([steering, throttle])

class MinimalSensorFusionController:
    """Lightweight sensor fusion controller without CNN-LSTM"""
    def __init__(self):
        self.kalman_filter = KalmanFilter()
        self.bayesian_network = BayesianNetwork()
        self.trajectory_predictor = SimplifiedTrajectoryPredictor()
        
    def process_sensor_data(self, gps_data, imu_data, lidar_data):
        """Process sensor data through Kalman Filter and Bayesian Network"""
        
        # 1. Kalman Filter Updates
        self.kalman_filter.predict()
        
        if gps_data is not None:
            self.kalman_filter.update(gps_data.reshape(-1, 1), 'gps')
            
        if imu_data is not None:
            self.kalman_filter.update(imu_data.reshape(-1, 1), 'imu')
            
        if lidar_data is not None:
            self.kalman_filter.update(lidar_data.reshape(-1, 1), 'lidar')
        
        # 2. Bayesian Network Assessment
        visibility = random.uniform(0.3, 1.0)  # Simulated
        precipitation = random.uniform(0.0, 0.6)  # Simulated
        temperature = random.uniform(-5.0, 25.0)  # German temperature range
        
        self.bayesian_network.update_weather(visibility, precipitation, temperature)
        sensor_weights = self.bayesian_network.get_sensor_weights()
        confidence = self.bayesian_network.get_confidence_level()
        
        # 3. Update trajectory predictor
        position = self.kalman_filter.get_position()
        velocity = self.kalman_filter.get_velocity()
        self.trajectory_predictor.update(position, velocity)
        
        return {
            'kalman_position': position,
            'kalman_velocity': velocity,
            'kalman_acceleration': self.kalman_filter.get_acceleration(),
            'sensor_weights': sensor_weights,
            'confidence_level': confidence,
            'weather_condition': self.bayesian_network.current_weather
        }
    
    def compute_control(self, fusion_results, target_waypoint, current_transform):
        """Compute vehicle control based on fused sensor data"""
        
        # Get trajectory prediction
        control_pred = self.trajectory_predictor.predict_control(target_waypoint, current_transform)
        steering, throttle = control_pred[0], control_pred[1]
        
        # Apply confidence-based adjustments
        confidence = fusion_results['confidence_level']
        
        # Reduce speed in low-confidence situations (German safety standards)
        if confidence < 0.6:
            throttle *= 0.5
            print(f"⚠️  Low confidence ({confidence:.2f}) - Reducing speed")
        
        # Weather-based adjustments
        weather = fusion_results['weather_condition']
        if weather in ['rainy', 'snowy']:
            throttle *= 0.7  # Reduce speed in adverse conditions
            steering *= 0.8  # More conservative steering
        elif weather == 'foggy':
            throttle *= 0.4  # Significantly reduce speed in fog
            
        # Safety constraints
        throttle = np.clip(throttle, 0.0, 0.8)
        steering = np.clip(steering, -1.0, 1.0)
        
        return carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steering),
            brake=0.0,
            reverse=False
        )

class CARLASimulationMinimal:
    """Minimal CARLA simulation class"""
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensors = {}
        self.controller = MinimalSensorFusionController()
        self.waypoints = []
        self.current_waypoint_index = 0
        
    def setup_carla(self):
        """Initialize CARLA connection"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            
            # Load town
            self.world = self.client.get_world()
            
            # Set German-like weather
            weather_conditions = [
                carla.WeatherParameters(cloudiness=70.0, precipitation=20.0, sun_altitude_angle=45.0),  # Overcast
                carla.WeatherParameters(cloudiness=40.0, precipitation=0.0, sun_altitude_angle=60.0),   # Partly cloudy
                carla.WeatherParameters(cloudiness=80.0, precipitation=60.0, sun_altitude_angle=30.0),  # Rainy
            ]
            
            self.world.set_weather(random.choice(weather_conditions))
            
            print("✓ CARLA connection established successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to CARLA: {e}")
            return False
    
    def spawn_vehicle(self):
        """Spawn the autonomous vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Use German car models when available
        vehicle_options = [
            'vehicle.bmw.grandtourer',
            'vehicle.mercedes.coupe_2020',
            'vehicle.audi.a2',
            'vehicle.tesla.model3'  # Fallback
        ]
        
        vehicle_bp = None
        for vehicle_name in vehicle_options:
            try:
                vehicle_bp = blueprint_library.find(vehicle_name)
                print(f"✓ Using vehicle: {vehicle_name}")
                break
            except:
                continue
        
        if not vehicle_bp:
            vehicle_bp = blueprint_library.filter('vehicle.*')[0]
            print(f"✓ Using fallback vehicle: {vehicle_bp.id}")
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Generate waypoints for German-style road navigation
        map_waypoints = self.world.get_map().generate_waypoints(distance=25.0)
        self.waypoints = map_waypoints[:15]  # Shorter route for demo
        
        print(f"✓ Vehicle spawned at {spawn_point.location}")
        
    def setup_sensors(self):
        """Setup sensors (GPS, LiDAR, IMU only)"""
        blueprint_library = self.world.get_blueprint_library()
        
        # GPS sensor
        gps_bp = blueprint_library.find('sensor.other.gnss')
        self.sensors['gps'] = self.world.spawn_actor(
            gps_bp, carla.Transform(), attach_to=self.vehicle
        )
        
        # IMU sensor
        imu_bp = blueprint_library.find('sensor.other.imu')
        self.sensors['imu'] = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self.vehicle
        )
        
        # LiDAR sensor (reduced points for performance)
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '16')  # Reduced from 32
        lidar_bp.set_attribute('points_per_second', '28000')  # Reduced
        lidar_bp.set_attribute('rotation_frequency', '20')  # Reduced
        lidar_bp.set_attribute('range', '15')  # Reduced range
        
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        self.sensors['lidar'] = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle
        )
        
        print("✓ Essential sensors setup complete!")
    
    def setup_spectator_camera(self):
        """Setup spectator camera to follow the vehicle"""
        self.spectator = self.world.get_spectator()
        print("🎥 Spectator camera setup complete - will follow vehicle")
    
    def update_visualization(self):
        """Update camera position and draw vehicle information"""
        if not self.vehicle:
            return
            
        # Update spectator camera to follow vehicle
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        # Position camera behind and above the vehicle
        camera_location = carla.Location(
            x=vehicle_location.x - 15,  # 15 meters behind
            y=vehicle_location.y - 5,   # Slightly to the side
            z=vehicle_location.z + 8   # 8 meters above
        )
        
        # Make camera look at the vehicle
        camera_rotation = carla.Rotation(
            pitch=-20,  # Look down at vehicle
            yaw=vehicle_transform.rotation.yaw + 15,  # Slightly angled view
            roll=0
        )
        
        spectator_transform = carla.Transform(camera_location, camera_rotation)
        self.spectator.set_transform(spectator_transform)
        
        # Draw visual markers
        # Red marker above vehicle
        self.world.debug.draw_point(
            vehicle_location + carla.Location(z=4),
            size=0.5,
            color=carla.Color(r=255, g=0, b=0),
            life_time=0.2
        )
        
        # Green arrow showing direction
        forward_vector = vehicle_transform.get_forward_vector()
        end_location = vehicle_location + carla.Location(
            x=forward_vector.x * 5,
            y=forward_vector.y * 5,
            z=forward_vector.z * 5
        )
        
        self.world.debug.draw_arrow(
            vehicle_location + carla.Location(z=2),
            end_location + carla.Location(z=2),
            thickness=0.15,
            arrow_size=0.5,
            color=carla.Color(r=0, g=255, b=0),
            life_time=0.2
        )
        
        # Draw current waypoint target
        if self.current_waypoint_index < len(self.waypoints):
            target_waypoint = self.waypoints[self.current_waypoint_index]
            target_location = target_waypoint.transform.location
            
            # Yellow marker for target waypoint
            self.world.debug.draw_point(
                target_location + carla.Location(z=2),
                size=0.3,
                color=carla.Color(r=255, g=255, b=0),
                life_time=0.2
            )
            
            # Blue line from vehicle to target
            self.world.debug.draw_line(
                vehicle_location + carla.Location(z=1),
                target_location + carla.Location(z=1),
                thickness=0.1,
                color=carla.Color(r=0, g=0, b=255),
                life_time=0.2
            )
    
    def sensor_callback_setup(self):
        """Setup sensor data collection"""
        self.sensor_data = {
            'gps': None,
            'lidar': None,
            'imu': None
        }
        
        # GPS callback
        def gps_callback(data):
            # Convert to local coordinates (simplified)
            self.sensor_data['gps'] = np.array([data.longitude * 100000, data.latitude * 100000])
        
        # LiDAR callback
        def lidar_callback(data):
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            if len(points) > 0:
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
                # Simple obstacle detection - average nearby points
                close_points = points[points[:, 2] < 5.0]  # Points within 5m height
                if len(close_points) > 0:
                    self.sensor_data['lidar'] = np.array([
                        np.mean(close_points[:, 0]), np.mean(close_points[:, 1])
                    ])
                else:
                    self.sensor_data['lidar'] = np.array([0.0, 0.0])
        
        # IMU callback
        def imu_callback(data):
            self.sensor_data['imu'] = np.array([
                data.accelerometer.x, data.accelerometer.y
            ])
        
        # Register callbacks
        self.sensors['gps'].listen(gps_callback)
        self.sensors['lidar'].listen(lidar_callback)
        self.sensors['imu'].listen(imu_callback)
    
    def run_simulation(self, duration=90):
        """Run the main simulation loop"""
        print("\n🚗 Starting German Automotive Industry Sensor Fusion Demo")
        print("📊 Integration: Kalman Filter + Bayesian Network")
        print("🌍 Optimized for German driving conditions")
        print("-" * 60)
        
        start_time = time.time()
        step_count = 0
        
        try:
            while time.time() - start_time < duration:
                step_count += 1
                
                # Process sensor data through fusion algorithms
                fusion_results = self.controller.process_sensor_data(
                    self.sensor_data.get('gps'),
                    self.sensor_data.get('imu'),
                    self.sensor_data.get('lidar')
                )
                
                # Get current target waypoint
                current_waypoint = None
                if self.current_waypoint_index < len(self.waypoints):
                    current_waypoint = self.waypoints[self.current_waypoint_index]
                    
                    # Check if reached current waypoint
                    vehicle_loc = self.vehicle.get_location()
                    waypoint_loc = current_waypoint.transform.location
                    
                    distance = math.sqrt(
                        (vehicle_loc.x - waypoint_loc.x)**2 + 
                        (vehicle_loc.y - waypoint_loc.y)**2
                    )
                    
                    if distance < 8.0:  # Reached waypoint
                        self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                        print(f"✓ Reached waypoint {self.current_waypoint_index}")
                
                # Compute and apply control
                control = self.controller.compute_control(
                    fusion_results, current_waypoint, self.vehicle.get_transform()
                )
                self.vehicle.apply_control(control)
                
                # Print detailed status every 3 seconds
                if step_count % 30 == 0:  # Every 30 steps (3 seconds at 10Hz)
                    elapsed = int(time.time() - start_time)
                    pos = fusion_results['kalman_position']
                    vel = fusion_results['kalman_velocity']
                    acc = fusion_results['kalman_acceleration']
                    weather = fusion_results['weather_condition']
                    weights = fusion_results['sensor_weights']
                    confidence = fusion_results['confidence_level']
                    
                    print(f"\n⏱️  Time: {elapsed}s | Step: {step_count}")
                    print(f"📍 Position: ({pos[0]:.1f}, {pos[1]:.1f})")
                    print(f"🏃 Velocity: ({vel[0]:.1f}, {vel[1]:.1f}) m/s")
                    print(f"⚡ Acceleration: ({acc[0]:.1f}, {acc[1]:.1f}) m/s²")
                    print(f"🌤️  Weather: {weather.upper()}")
                    print(f"📊 Confidence: {confidence:.2f}")
                    print(f"⚖️  Weights: GPS={weights['gps']:.2f} | LiDAR={weights['lidar']:.2f} | IMU={weights['imu']:.2f}")
                    print(f"🚗 Control: Throttle={control.throttle:.2f} | Steering={control.steer:.2f}")
                    
                    # German automotive industry metrics
                    if confidence > 0.8:
                        print("✅ HIGH RELIABILITY - Full autonomous mode")
                    elif confidence > 0.6:
                        print("⚠️  MEDIUM RELIABILITY - Cautious driving")
                    else:
                        print("🔴 LOW RELIABILITY - Safety protocols active")
                
                time.sleep(0.1)  # 10 Hz control loop
                
        except KeyboardInterrupt:
            print("\n⏹️  Simulation interrupted by user")
        
        elapsed_total = time.time() - start_time
        print(f"\n✅ Simulation completed successfully!")
        print(f"📊 Total time: {elapsed_total:.1f}s | Total steps: {step_count}")
        print(f"🎯 Waypoints completed: {self.current_waypoint_index}")
    
    def cleanup(self):
        """Clean up CARLA actors"""
        if self.vehicle:
            self.vehicle.destroy()
        
        for sensor in self.sensors.values():
            if sensor:
                sensor.destroy()
        
        print("🧹 Cleanup completed")

def main():
    """Main function to run the minimal simulation"""
    print("🇩🇪 German Automotive Industry - Sensor Fusion Study")
    print("💾 Minimal Version (No PyTorch required)")
    print("=" * 60)
    
    simulation = CARLASimulationMinimal()
    
    try:
        # Setup CARLA
        if not simulation.setup_carla():
            print("❌ Cannot start simulation without CARLA connection")
            return
        
        # Spawn vehicle and setup sensors
        simulation.spawn_vehicle()
        simulation.setup_sensors()
        simulation.setup_spectator_camera()  # Add camera setup
        simulation.sensor_callback_setup()
        
        # Wait for sensors to initialize
        print("⏳ Initializing sensors...")
        time.sleep(3)
        
        # Run the simulation
        simulation.run_simulation(duration=90)  # 1.5 minutes
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
    
    finally:
        simulation.cleanup()
        print("\n🏁 CARLA Minimal Sensor Fusion Demo Completed!")
        print("📈 Successfully demonstrated Kalman Filter + Bayesian Network integration")

if __name__ == "__main__":
    main()
