import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import serial
import serial.tools.list_ports
import time


# === GLOBAL PARAMETERS (update as needed) === (All units in cm)
SERVO_HORN_LENGTH = 8.0  # |h|, length from servo shaft to hinge (servo horn)
ROD_LENGTH = 9.4         # |d|, length from hinge to plate anchor (rod)
PLATE_RADIUS = 14.0        # pr, radius of the circular plate
BASE_RADIUS = 5.0        # br, radius of the base circle

# Per-servo zero-angle calibration offsets (in degrees).
# Positive values mean the physical "zero" sits at +offset degrees and you must add the offset
# to the computed angle before commanding the servo. Tune these per motor.
SERVO_ZERO_OFFSETS_DEG = np.full(3, 15.5)

class StewartPlatform:
	def __init__(self):
		self.num_servos = 3
		self.servo_angles = np.zeros(3)
		# Calculate base and plate anchor points (circular arrangement)
		self.base_points = self._calculate_circle_points(BASE_RADIUS, 3, z=0)
		self.plate_points = self._calculate_circle_points(PLATE_RADIUS, 3, z=0)
		# For each leg, calculate the servo horn orientation (beta_k)
		self.servo_betas = self._calculate_servo_betas(3)
		# Calculate the optimal neutral height (where horn and rod are orthogonal)
		self.neutral_height = self._calculate_neutral_height()
		# Copy of zero-angle offsets (deg) used to bias command outputs
		self.servo_zero_offsets_deg = SERVO_ZERO_OFFSETS_DEG.copy()

	def _calculate_circle_points(self, radius, num_points, z=0, start_angle=0.0, ccw=True):
		"""
		Return points evenly spaced on a circle in the XY plane.

		Args:
			radius: circle radius
			num_points: number of points
			z: z coordinate for all points
			start_angle: angle (radians) of the first point measured from +X axis
			ccw: if True angles increase counter-clockwise (positive rotation),
			     otherwise they increase clockwise.

		This makes the coordinate convention explicit: start_angle=0 aligns the
		first motor with +X, and ccw=True means sweeping from +X towards +Y will
		reach the second motor (motor 1) in the positive rotation direction.
		"""
		sign = 1.0 if ccw else -1.0
		angles = start_angle + sign * np.linspace(0, 2 * np.pi, num_points, endpoint=False)
		return np.array([
			[radius * np.cos(a), radius * np.sin(a), z] for a in angles
		])

	def _calculate_servo_betas(self, num_servos):
		# Derive servo beta angles from the base anchor points so the betas
		# always match the actual base point geometry and the chosen start_angle/ccw.
		# beta_k is the outward radial direction angle measured from +X.
		xy = self.base_points[:, :2]
		betas = np.arctan2(xy[:,1], xy[:,0])
		return betas
	
	def _calculate_neutral_height(self):
		# Calculate optimal height where horn and rod are orthogonal (90 degrees)
		# From article: z = sqrt(d^2 + h^2 - (p_k_x - b_k_x)^2 - (p_k_y - b_k_y)^2)
		# Use the first leg for calculation (all should give same result for circular arrangement)
		pk = self.plate_points[0]
		bk = self.base_points[0]
		planar_dist_sq = (pk[0] - bk[0])**2 + (pk[1] - bk[1])**2
		z_sq = ROD_LENGTH**2 + SERVO_HORN_LENGTH**2 - planar_dist_sq
		if z_sq < 0:
			print(f"Warning: Cannot reach neutral position with current geometry!")
			print(f"  Planar distance: {np.sqrt(planar_dist_sq):.2f}")
			print(f"  Max reach: {np.sqrt(ROD_LENGTH**2 + SERVO_HORN_LENGTH**2):.2f}")
			return 10.0  # fallback
		# Add extra height offset to raise platform higher
		return np.sqrt(z_sq) + 3.0

	def calculate_servo_angles(self, plate_normal, translation=None, *, degrees=False, apply_offsets=False, clamp_min=None, clamp_max=None):
		"""
		Calculate the servo angles (alpha_k) for a given plate orientation (normal) and translation.
		Args:
			plate_normal: 3D normal vector for the plate
			translation: 3D translation of the plate center (default: neutral height)
		Returns:
			Array of servo angles (radians)
		"""
		if translation is None:
			translation = np.array([0, 0, self.neutral_height])
		
		n = np.array(plate_normal)
		n = n / np.linalg.norm(n)
		# Find rotation matrix to align [0,0,1] to n
		z_axis = np.array([0,0,1])
		v = np.cross(z_axis, n)
		c = np.dot(z_axis, n)
		if np.linalg.norm(v) < 1e-8:
			R = np.eye(3)
		else:
			vx = np.array([[0, -v[2], v[1]],
						  [v[2], 0, -v[0]],
						  [-v[1], v[0], 0]])
			R = np.eye(3) + vx + vx @ vx * ((1-c)/(np.linalg.norm(v)**2))
		# Platform anchor points in world frame
		plate_points_world = (R @ self.plate_points.T).T + translation
		# For each leg, calculate the required servo angle
		for k in range(self.num_servos):
			Bk = self.base_points[k]
			Pk = plate_points_world[k]
			lk = Pk - Bk
			beta_k = self.servo_betas[k]
			# e_k, f_k, g_k as per article
			e_k = 2 * SERVO_HORN_LENGTH * lk[2]
			f_k = 2 * SERVO_HORN_LENGTH * (np.cos(beta_k) * lk[0] + np.sin(beta_k) * lk[1])
			g_k = np.dot(lk, lk) - (ROD_LENGTH**2 - SERVO_HORN_LENGTH**2)
			denom = np.sqrt(e_k**2 + f_k**2)
			# Clamp for arcsin domain
			arg = np.clip(g_k / denom, -1.0, 1.0)
			alpha_k = np.arcsin(arg) - np.arctan2(f_k, e_k)
			self.servo_angles[k] = alpha_k

		# Optionally apply zero offsets and clamping, returning degrees
		if apply_offsets or (clamp_min is not None and clamp_max is not None) or degrees:
			angles_deg = np.degrees(self.servo_angles)
			if apply_offsets:
				angles_deg = angles_deg + self.servo_zero_offsets_deg
			if clamp_min is not None and clamp_max is not None:
				angles_deg = np.clip(angles_deg, clamp_min, clamp_max)
			return angles_deg

		return self.servo_angles

	def get_platform_points(self, plate_normal, translation=None):
		if translation is None:
			translation = np.array([0, 0, self.neutral_height])
		
		n = np.array(plate_normal)
		n = n / np.linalg.norm(n)
		z_axis = np.array([0,0,1])
		v = np.cross(z_axis, n)
		c = np.dot(z_axis, n)
		if np.linalg.norm(v) < 1e-8:
			R = np.eye(3)
		else:
			vx = np.array([[0, -v[2], v[1]],
						  [v[2], 0, -v[0]],
						  [-v[1], v[0], 0]])
			R = np.eye(3) + vx + vx @ vx * ((1-c)/(np.linalg.norm(v)**2))
		plate_points_world = (R @ self.plate_points.T).T + translation
		return self.base_points, plate_points_world


class StewartPlatformVisualizer:
	def __init__(self, platform: StewartPlatform, arduino_serial=None):
		self.platform = platform
		self.fig = None
		self.ax = None
		self.arduino_serial = arduino_serial

	def send_to_arduino(self, angles_deg):
		"""Send servo angles to Arduino in degrees"""
		if self.arduino_serial and self.arduino_serial.is_open:
			# Format: "angle0,angle1,angle2\n"
			command = f"{int(angles_deg[0])},{int(angles_deg[1])},{int(angles_deg[2])}\n"
			self.arduino_serial.write(command.encode())
			
			# Read response
			time.sleep(0.05)
			if self.arduino_serial.in_waiting:
				response = self.arduino_serial.readline().decode('utf-8', errors='ignore').strip()
				if response.startswith('OK:'):
					print(f"Arduino: {response}")
					return True
				elif response.startswith('ERR:'):
					print(f"Arduino error: {response}")
					return False
		return False

	def plot(self, plate_normal=[0,0,1], interactive=False):
		# Recalculate servo angles for the current orientation
		self.platform.calculate_servo_angles(plate_normal)
		base_points, plate_points = self.platform.get_platform_points(plate_normal)
		
		# Create figure if it doesn't exist, otherwise clear it
		if self.fig is None or not interactive:
			self.fig = plt.figure(figsize=(10, 8))
			self.ax = self.fig.add_subplot(111, projection='3d')
		else:
			self.ax.clear()
		
		# Plot base
		self.ax.scatter(base_points[:,0], base_points[:,1], base_points[:,2], c='b', label='Base', s=50)
		# Plot plate
		self.ax.scatter(plate_points[:,0], plate_points[:,1], plate_points[:,2], c='r', label='Plate', s=50)
		# Plot two-link structure for each leg
		for i in range(self.platform.num_servos):
			# Get base, plate, and servo angle
			Bk = base_points[i]
			Pk = plate_points[i]
			beta_k = self.platform.servo_betas[i]
			alpha_k = self.platform.servo_angles[i]
			# Servo horn endpoint (bearing location)
			# From article: H_k = B_k + |h| * (cos(alpha_k)cos(beta_k), cos(alpha_k)sin(beta_k), sin(alpha_k))
			# This comes from Rz(beta_k) * Ry(-alpha_k) * [|h|, 0, 0]
			horn_vec = SERVO_HORN_LENGTH * np.array([
				np.cos(alpha_k) * np.cos(beta_k),
				np.cos(alpha_k) * np.sin(beta_k),
				np.sin(alpha_k)
			])
			Hk = Bk + horn_vec
			# Plot servo horn (base to bearing)
			self.ax.plot([Bk[0], Hk[0]], [Bk[1], Hk[1]], [Bk[2], Hk[2]], 'c-', linewidth=2, label='Servo Horn' if i==0 else None)
			# Plot rod (bearing to platform)
			self.ax.plot([Hk[0], Pk[0]], [Hk[1], Pk[1]], [Hk[2], Pk[2]], 'g-', linewidth=2, label='Rod' if i==0 else None)
			# Plot bearing joint
			self.ax.scatter([Hk[0]], [Hk[1]], [Hk[2]], c='orange', s=30)
		# Draw plate circle
		theta = np.linspace(0, 2*np.pi, 100)
		circle = np.array([
			[PLATE_RADIUS * np.cos(t), PLATE_RADIUS * np.sin(t), 0] for t in theta
		])
		# Rotate plate circle
		n = np.array(plate_normal)
		n = n / np.linalg.norm(n)
		z_axis = np.array([0,0,1])
		v = np.cross(z_axis, n)
		c = np.dot(z_axis, n)
		if np.linalg.norm(v) < 1e-8:
			R = np.eye(3)
		else:
			vx = np.array([[0, -v[2], v[1]],
						  [v[2], 0, -v[0]],
						  [-v[1], v[0], 0]])
			R = np.eye(3) + vx + vx @ vx * ((1-c)/(np.linalg.norm(v)**2))
		circle_rot = (R @ circle.T).T + np.mean(plate_points, axis=0)
		self.ax.plot(circle_rot[:,0], circle_rot[:,1], circle_rot[:,2], 'r--', alpha=0.5)
		self.ax.set_xlabel('X')
		self.ax.set_ylabel('Y')
		self.ax.set_zlabel('Z')
		self.ax.legend()
		self.ax.set_box_aspect([1,1,1])
		
		# Set title with servo angles
		angles_deg = np.degrees(self.platform.servo_angles)
		self.ax.set_title(f'Stewart Platform\nServo Angles: [{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°, {angles_deg[2]:.1f}°]')
		
		# Send to Arduino if connected
		if self.arduino_serial:
			# Calculate command angles with offsets and clamping
			command_angles_deg = self.platform.calculate_servo_angles(
				plate_normal,
				degrees=True,
				apply_offsets=True,
				clamp_min=0.0,
				clamp_max=65.0
			)
			self.send_to_arduino(command_angles_deg)
		
		if interactive:
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
		else:
			plt.show()

def find_arduino_port():
	"""Auto-detect Arduino port"""
	ports = serial.tools.list_ports.comports()
	for p in ports:
		# Look for common Arduino USB identifiers
		if 'usbmodem' in p.device or 'usbserial' in p.device or 'Arduino' in str(p.description):
			print(f"Found Arduino on: {p.device}")
			return p.device
	return None

def connect_arduino(port=None, baud=115200):
	"""Connect to Arduino via serial"""
	if port is None:
		port = find_arduino_port()
	
	if not port:
		print("No Arduino found. Running in simulation mode only.")
		return None
	
	try:
		ser = serial.Serial(port, baud, timeout=1)
		time.sleep(2)  # Wait for Arduino to reset
		
		# Read startup messages
		while ser.in_waiting:
			line = ser.readline().decode('utf-8', errors='ignore').strip()
			print(f"Arduino: {line}")
		
		print(f"Connected to Arduino on {port} at {baud} baud")
		print("Initializing all servos to 0 degrees...")
		# Send all servos to 0 degrees directly
		command = "0,0,0\n"
		ser.write(command.encode())
		time.sleep(0.1)
		if ser.in_waiting:
			response = ser.readline().decode('utf-8', errors='ignore').strip()
			print(f"Arduino: {response}")
		print("Servos initialized: [0°, 0°, 0°]")
		return ser
		
	except serial.SerialException as e:
		print(f"Failed to connect to {port}: {e}")
		print("Running in simulation mode only.")
		return None

def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Stewart Platform Visualizer')
	parser.add_argument('--update-real', action='store_true', 
						help='Send servo angles to Arduino in real-time')
	parser.add_argument('--port', type=str, default=None,
						help='Arduino serial port (auto-detect if not specified)')
	parser.add_argument('--baud', type=int, default=115200,
						help='Serial baud rate (default: 115200)')
	args = parser.parse_args()
	
	# Connect to Arduino if requested
	arduino_serial = None
	if args.update_real:
		arduino_serial = connect_arduino(args.port, args.baud)
		if arduino_serial is None:
			print("Warning: Could not connect to Arduino. Continuing in simulation mode.")
	
	plt.ion()  # Enable interactive mode
	platform = StewartPlatform()
	
	# Initialize servos to 0 degrees manually
	platform.servo_angles = np.array([0.0, 0.0, 0.0])
	
	visualizer = StewartPlatformVisualizer(platform, arduino_serial)
	# Initial normal vector (vertical)
	normal = np.array([0, 0, 1], dtype=float)
	
	# Tilt limits (max deviation from vertical in radians)
	MAX_TILT = np.radians(30)  # 30 degrees max tilt

	def on_key(event):
		nonlocal normal
		step = 0.1
		if event.key == 'up':
			normal[1] += step
		elif event.key == 'down':
			normal[1] -= step
		elif event.key == 'left':
			normal[0] -= step
		elif event.key == 'right':
			normal[0] += step
		elif event.key == 'r':
			normal = np.array([0, 0, 1], dtype=float)
		else:
			return  # Ignore other keys
		
		# Normalize to avoid drift
		if np.linalg.norm(normal) < 1e-6:
			normal = np.array([0, 0, 1], dtype=float)
		else:
			normal = normal / np.linalg.norm(normal)
		
		# Apply tilt limits
		z_axis = np.array([0, 0, 1])
		tilt_angle = np.arccos(np.clip(np.dot(normal, z_axis), -1, 1))
		if tilt_angle > MAX_TILT:
			# Project back to max tilt
			# Find the direction of tilt in x-y plane
			tilt_direction = np.array([normal[0], normal[1], 0])
			if np.linalg.norm(tilt_direction) > 1e-6:
				tilt_direction = tilt_direction / np.linalg.norm(tilt_direction)
				# Construct normal at max tilt angle
				normal = np.array([
					tilt_direction[0] * np.sin(MAX_TILT),
					tilt_direction[1] * np.sin(MAX_TILT),
					np.cos(MAX_TILT)
				])
				normal = normal / np.linalg.norm(normal)
		
		# Update the plot
		visualizer.plot(normal, interactive=True)

	# Initial plot
	visualizer.plot(normal, interactive=True)
	# Connect key press event
	visualizer.fig.canvas.mpl_connect('key_press_event', on_key)
	
	try:
		plt.show(block=True)
	finally:
		# Clean up Arduino connection
		if arduino_serial and arduino_serial.is_open:
			print("\nClosing Arduino connection...")
			arduino_serial.close()

if __name__ == "__main__":
	main()