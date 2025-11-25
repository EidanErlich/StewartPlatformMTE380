import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Sequence
from numpy.typing import NDArray


# === GLOBAL PARAMETERS (update as needed) === (All units in cm)
SERVO_HORN_LENGTH = 8.0  # |h|, length from servo shaft to hinge (servo horn)
ROD_LENGTH = 9.4         # |d|, length from hinge to plate anchor (rod)
PLATE_RADIUS = 14.0        # pr, radius of the circular plate
BASE_RADIUS = 5.0        # br, radius of the base circle

# Per-servo zero-angle calibration offsets (in degrees).
# Positive values mean the physical "zero" sits at +offset degrees and you must add the offset
# to the computed angle before commanding the servo. Tune these per motor.
SERVO_ZERO_OFFSETS_DEG = np.full(3, 15.5)


def _calculate_circle_points(radius, num_points, z=0, start_angle=0.0, ccw=True):
    """Return points evenly spaced on a circle in the XY plane."""
    sign = 1.0 if ccw else -1.0
    angles = start_angle + sign * np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    return np.array([
        [radius * np.cos(a), radius * np.sin(a), z] for a in angles
    ])


def _calculate_neutral_height(bases, contacts, motor_len, push_len):
    """Calculate optimal height where horn and rod are orthogonal."""
    # Use first leg for calculation
    base = np.array(bases[0])
    contact = np.array([contacts[0][0], contacts[0][1], 0.0])
    planar_dist_sq = (contact[0] - base[0])**2 + (contact[1] - base[1])**2
    z_sq = push_len**2 + motor_len**2 - planar_dist_sq
    if z_sq < 0:
        return 10.0  # fallback
    return np.sqrt(z_sq) + 3.0


class Settings:
    """Settings class for inverse kinematics geometry parameters."""
    
    # Link lengths (cm)
    MOTOR_LINK_LEN = SERVO_HORN_LENGTH  # 8.0 cm
    PUSH_LINK_LEN = ROD_LENGTH  # 9.4 cm
    
    # Base points (3D coordinates in cm)
    BASES = _calculate_circle_points(BASE_RADIUS, 3, z=0).tolist()
    
    # Contact points on plate (2D local coordinates in cm, z=0 implied)
    CONTACTS = _calculate_circle_points(PLATE_RADIUS, 3, z=0)[:, :2].tolist()
    
    # Table height (cm) - calculated as neutral height
    TABLE_HEIGHT = _calculate_neutral_height(BASES, CONTACTS, MOTOR_LINK_LEN, PUSH_LINK_LEN)


def rotation_matrix(roll: float, pitch: float) -> NDArray[np.float64]:
    """Compute rotation matrix from roll and pitch angles."""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return R_y @ R_x  # type: ignore


def bearing_point_exact(
    base: Sequence[float],
    p_world: Sequence[float],
    l1: float,
    l2: float,
) -> Tuple[bool, Optional[NDArray[np.float64]]]:
    """Calculate bearing point for a two-link mechanism."""
    B = np.array(base, dtype=np.float64)
    P = np.array(p_world, dtype=np.float64)
    O = np.array([0.0, 0.0, 0.0])
    d = np.linalg.norm(P - B)
    if d > l1 + l2 + 1e-9 or d < abs(l1 - l2) - 1e-9 or d < 1e-9:
        return False, None
    v = (P - B) / d
    a = (l1**2 - l2**2 + d**2) / (2.0 * d)
    h2 = l1**2 - a**2
    if h2 < -1e-12:
        return False, None
    h2 = max(h2, 0.0)
    h = np.sqrt(h2)
    M = B + a * v
    # Define radial direction in XY plane
    radial_dir = B - O
    radial_dir[2] = 0.0
    norm = np.linalg.norm(radial_dir)
    if norm < 1e-9:
        radial_dir = np.array([1.0, 0.0, 0.0])
    else:
        radial_dir /= norm
    # Plane normal = cross(radial, Z)
    plane_normal = np.cross(radial_dir, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(plane_normal) < 1e-9:
        plane_normal = np.array([0.0, 1.0, 0.0])
    else:
        plane_normal /= np.linalg.norm(plane_normal)
    # Perpendicular in plane
    n_perp = np.cross(v, plane_normal)
    n_perp /= np.linalg.norm(n_perp)
    bearing1 = M + h * n_perp
    bearing2 = M - h * n_perp
    bearing = bearing1 if bearing1[2] < bearing2[2] else bearing2
    return True, bearing


def motor_angle_deg(base, bearing):
    """Calculate motor angle in degrees from base and bearing points."""
    base = np.array(base, dtype=float)
    bearing = np.array(bearing, dtype=float)
    L = bearing - base
    x, y, z = L
    angle_rad = np.arctan2(z, np.hypot(x, y))
    return float(np.rad2deg(angle_rad))


def leg_points_rigid(
    base: NDArray[np.float64],
    contact_local: NDArray[np.float64],
    plane_pose: Tuple[float, float, float],
    l1: float,
    l2: float,
) -> Optional[
    Tuple[
        Tuple[NDArray[np.float64], NDArray[np.float64]],
        Tuple[NDArray[np.float64], NDArray[np.float64]],
    ]
]:
    """Calculate leg points for rigid plate."""
    roll, pitch, z = plane_pose
    R = rotation_matrix(roll, pitch)
    P_world = R @ np.array([contact_local[0], contact_local[1], 0.0]) + np.array([0, 0, z])
    ok, bearing = bearing_point_exact(base, P_world, l1, l2)
    if not ok:
        return None
    return (base.copy(), bearing), (bearing.copy(), P_world.copy())


def solve_motor_angles_for_plane(roll_deg, pitch_deg, z=Settings.TABLE_HEIGHT):
    """Solve motor angles for given roll and pitch angles."""
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    out = np.full(3, np.nan)
    for i, (b, c) in enumerate(zip(Settings.BASES, Settings.CONTACTS)):
        leg_points = leg_points_rigid(
            np.array(b), 
            np.array([c[0], c[1], 0.0]), 
            (roll, pitch, z), 
            Settings.MOTOR_LINK_LEN, 
            Settings.PUSH_LINK_LEN
        )
        if leg_points:
            leg_1_points, _ = leg_points
            base_point, bearing_point = leg_1_points
            out[i] = motor_angle_deg(base_point, bearing_point)
    return out


def normal_to_roll_pitch(normal: np.ndarray) -> Tuple[float, float]:
    """
    Convert normal vector to roll and pitch angles (in degrees).
    
    Args:
        normal: Unit normal vector [nx, ny, nz]
        
    Returns:
        (roll_deg, pitch_deg): Roll and pitch angles in degrees
        
    Notes:
        The rotation matrix is R = R_y(pitch) @ R_x(roll) applied to [0, 0, 1].
        This gives: n = [sin(pitch)*cos(roll), -sin(roll), cos(pitch)*cos(roll)]
        Therefore:
        - roll = -arcsin(ny)
        - pitch = arcsin(nx / cos(roll)) = arcsin(nx / sqrt(1 - ny^2))
    """
    n = normal / np.linalg.norm(normal)
    
    # Extract components
    nx, ny, nz = n[0], n[1], n[2]
    
    # Ensure nz is positive (platform facing up)
    if nz < 0:
        n = -n
        nx, ny, nz = n[0], n[1], n[2]
    
    # Convert to roll/pitch
    # From R_y(pitch) @ R_x(roll) @ [0, 0, 1]:
    # nx = sin(pitch) * cos(roll)
    # ny = -sin(roll)
    # nz = cos(pitch) * cos(roll)
    
    # Extract roll first: roll = -arcsin(ny)
    roll_rad = -np.arcsin(np.clip(ny, -1.0, 1.0))
    cos_roll = np.cos(roll_rad)
    
    # Extract pitch: pitch = arcsin(nx / cos(roll))
    if abs(cos_roll) < 1e-6:
        # Near 90 degree roll, use alternative method
        # In this case, nz ≈ 0, so we can use atan2
        pitch_rad = np.arctan2(nx, nz)
    else:
        pitch_rad = np.arcsin(np.clip(nx / cos_roll, -1.0, 1.0))
    
    return float(np.rad2deg(roll_rad)), float(np.rad2deg(pitch_rad))


class StewartPlatform:
    """Stewart Platform inverse kinematics using new implementation."""
    
    def __init__(self):
        self.num_servos = 3
        self.servo_angles = np.zeros(3)
        # Calculate base and plate anchor points (circular arrangement)
        self.base_points = self._calculate_circle_points(BASE_RADIUS, 3, z=0)
        self.plate_points = self._calculate_circle_points(PLATE_RADIUS, 3, z=0)
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
    
    def _calculate_neutral_height(self):
        """Calculate optimal height where horn and rod are orthogonal (90 degrees)."""
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
            degrees: If True, return angles in degrees; otherwise radians
            apply_offsets: If True, apply servo zero offsets
            clamp_min: Minimum angle (degrees, only used if degrees=True)
            clamp_max: Maximum angle (degrees, only used if degrees=True)
            
        Returns:
            Array of servo angles (radians or degrees)
        """
        if translation is None:
            translation = np.array([0, 0, self.neutral_height])
        
        # Normalize input normal vector
        n = np.array(plate_normal, dtype=float)
        n = n / np.linalg.norm(n)
        
        # Ensure nz is positive (platform facing up)
        if n[2] < 0:
            n = -n
        
        # Convert normal vector to roll/pitch angles
        roll_deg, pitch_deg = normal_to_roll_pitch(n)
        
        # Get z height from translation
        z = translation[2] if len(translation) >= 3 else self.neutral_height
        
        # Solve for motor angles using new implementation
        angles_deg = solve_motor_angles_for_plane(roll_deg, pitch_deg, z)
        
        # Check for NaN values (infeasible configuration)
        if np.any(np.isnan(angles_deg)):
            # Fallback: return zeros or previous angles
            angles_deg = np.zeros(3)
        
        # Store angles in radians for internal state
        self.servo_angles = np.deg2rad(angles_deg)
        
        # Optionally apply zero offsets and clamping, returning degrees
        if apply_offsets or (clamp_min is not None and clamp_max is not None) or degrees:
            if apply_offsets:
                angles_deg = angles_deg + self.servo_zero_offsets_deg
            if clamp_min is not None and clamp_max is not None:
                angles_deg = np.clip(angles_deg, clamp_min, clamp_max)
            return angles_deg
        
        return self.servo_angles
    
    def get_platform_points(self, plate_normal, translation=None):
        """
        Get platform points for visualization.
        
        Args:
            plate_normal: 3D normal vector for the plate
            translation: 3D translation of the plate center (default: neutral height)
            
        Returns:
            (base_points, plate_points_world): Base points and transformed plate points
        """
        if translation is None:
            translation = np.array([0, 0, self.neutral_height])
        
        # Normalize input normal vector
        n = np.array(plate_normal, dtype=float)
        n = n / np.linalg.norm(n)
        
        # Ensure nz is positive (platform facing up)
        if n[2] < 0:
            n = -n
        
        # Convert normal vector to roll/pitch angles
        roll_deg, pitch_deg = normal_to_roll_pitch(n)
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        
        # Get z height from translation
        z = translation[2] if len(translation) >= 3 else self.neutral_height
        
        # Calculate rotation matrix
        R = rotation_matrix(roll, pitch)
        
        # Transform plate points to world frame
        plate_points_world = (R @ self.plate_points.T).T + translation
        
        return self.base_points, plate_points_world


class StewartPlatformVisualizer:
    """Visualizer for Stewart Platform."""
    
    def __init__(self, platform: StewartPlatform):
        self.platform = platform
        self.fig = None
        self.ax = None
    
    def plot(self, plate_normal=[0,0,1], interactive=False):
        """Plot the Stewart platform configuration."""
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
        # We need to calculate bearing points for visualization
        roll_deg, pitch_deg = normal_to_roll_pitch(np.array(plate_normal))
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        z = self.platform.neutral_height
        
        for i in range(self.platform.num_servos):
            # Get base and plate points
            Bk = base_points[i]
            Pk = plate_points[i]
            
            # Calculate bearing point using new implementation
            base_arr = Settings.BASES[i]
            contact_arr = Settings.CONTACTS[i]
            leg_points = leg_points_rigid(
                np.array(base_arr),
                np.array([contact_arr[0], contact_arr[1], 0.0]),
                (roll, pitch, z),
                Settings.MOTOR_LINK_LEN,
                Settings.PUSH_LINK_LEN
            )
            
            if leg_points:
                leg_1_points, _ = leg_points
                _, Hk = leg_1_points
            else:
                # Fallback: approximate bearing point
                Hk = (Bk + Pk) / 2
            
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
        roll_deg, pitch_deg = normal_to_roll_pitch(n)
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        R = rotation_matrix(roll, pitch)
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
        
        if interactive:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            plt.show()


def main():
    """Main function for interactive visualization."""
    plt.ion()  # Enable interactive mode
    platform = StewartPlatform()
    
    # Initialize servos to 0 degrees manually
    platform.servo_angles = np.array([0.0, 0.0, 0.0])
    
    visualizer = StewartPlatformVisualizer(platform)
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
    
    plt.show(block=True)


if __name__ == "__main__":
    main()
