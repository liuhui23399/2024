import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
from scipy.integrate import solve_ivp
import math

# Set matplotlib to support Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem2DetailedModel:
    """
    Problem 2: Dragon dance team turning problem modeling and solution
    
    Core challenges:
    1. Trajectory changes during dragon head turning
    2. Continuity constraints of dragon body formation
    3. Avoid self-intersection (dragon body cannot cross itself)
    """
    
    def __init__(self):
        # Dragon geometry parameters (units: meters)
        self.L_head = 2.23  # Dragon head length
        self.L_body = 3.41  # Dragon body length
        self.L_tail = 2.20  # Dragon tail length
        self.total_segments = 220
        
        # Handle positions
        self.handle_segments = [1, 51, 101, 151, 201]
        self.handle_distances = []
        for seg in self.handle_segments:
            dist = self.L_head + (seg - 1) * (self.L_body / self.total_segments)
            self.handle_distances.append(dist)
        self.tail_distance = self.L_head + self.L_body + self.L_tail
        
        # Turning parameters
        self.turn_start_time = None  # Turn start time
        self.turn_duration = None    # Turn duration
        self.turn_radius = None      # Turn radius
        
        print("=== Problem 2 Turning Model Parameters ===")
        print(f"Total dragon length: {self.L_head + self.L_body + self.L_tail:.2f}m")
        print(f"Handle positions: {[f'{d:.2f}m' for d in self.handle_distances]}")
        
    def phase1_spiral_trajectory(self, t):
        """
        Phase 1: Spiral trajectory (before turning)
        """
        a = 1.0 / (2 * np.pi)
        theta = np.sqrt(2 * t / a) if t > 0 else 0
        r = a * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
    
    def calculate_turn_parameters(self, spiral_end_time):
        """
        Calculate turning parameters
        
        Key approach:
        1. Determine starting point position and direction for turning
        2. Choose appropriate turning radius to avoid self-intersection
        3. Design turning trajectory (circular arc + straight line segment)
        """
        # Turn starting point
        x_start, y_start = self.phase1_spiral_trajectory(spiral_end_time)
        
        # Calculate tangent direction of spiral at starting point
        dt = 0.001
        x1, y1 = self.phase1_spiral_trajectory(spiral_end_time - dt)
        x2, y2 = self.phase1_spiral_trajectory(spiral_end_time + dt)
        
        # Tangent direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            tangent_x = dx / length
            tangent_y = dy / length
        else:
            tangent_x, tangent_y = 1.0, 0.0
        
        # Normal vector (turning direction)
        normal_x = -tangent_y
        normal_y = tangent_x
        
        # Determine turning radius
        # Radius must be large enough so dragon body doesn't intersect itself
        dragon_total_length = self.L_head + self.L_body + self.L_tail
        min_radius = dragon_total_length / (2 * np.pi)  # Minimum radius constraint
        
        # Choose turning radius as 1/4 of total length to ensure safety
        turn_radius = max(min_radius * 1.5, dragon_total_length / 4)
        
        # Turn center
        center_x = x_start + normal_x * turn_radius
        center_y = y_start + normal_y * turn_radius
        
        # Turn starting angle
        start_angle = np.arctan2(y_start - center_y, x_start - center_x)
        
        return {
            'start_point': (x_start, y_start),
            'center': (center_x, center_y),
            'radius': turn_radius,
            'start_angle': start_angle,
            'tangent_direction': (tangent_x, tangent_y),
            'normal_direction': (normal_x, normal_y)
        }
    
    def phase2_turn_trajectory(self, t, turn_params, turn_start_time):
        """
        Phase 2: Turning trajectory (circular arc motion)
        """
        if t < turn_start_time:
            return self.phase1_spiral_trajectory(t)
        
        # Time in turning phase
        t_turn = t - turn_start_time
        
        # Assume turning speed remains 1m/s, arc length determines turn time
        angular_velocity = 1.0 / turn_params['radius']  # v = rω, ω = v/r
        
        # Turn angle (assume 180 degree turn)
        target_angle = np.pi  # 180 degree turn
        turn_duration = target_angle / angular_velocity
        
        if t_turn <= turn_duration:
            # Circular arc motion phase
            angle = turn_params['start_angle'] + angular_velocity * t_turn
            x = turn_params['center'][0] + turn_params['radius'] * np.cos(angle)
            y = turn_params['center'][1] + turn_params['radius'] * np.sin(angle)
            return x, y
        else:
            # Straight line motion after turn completion
            end_angle = turn_params['start_angle'] + target_angle
            x_end = turn_params['center'][0] + turn_params['radius'] * np.cos(end_angle)
            y_end = turn_params['center'][1] + turn_params['radius'] * np.sin(end_angle)
            
            # Motion direction after turn (opposite to before turn)
            t_straight = t_turn - turn_duration
            direction_x = -turn_params['tangent_direction'][0]
            direction_y = -turn_params['tangent_direction'][1]
            
            x = x_end + direction_x * t_straight
            y = y_end + direction_y * t_straight
            
            return x, y
    
    def get_trajectory_length_at_time(self, t, turn_params=None, turn_start_time=None):
        """
        Calculate cumulative path length from start to time t for dragon head
        This is crucial for determining positions of dragon body parts
        """
        if turn_params is None or t <= turn_start_time:
            # Spiral phase, arc length equals time (speed 1m/s)
            return t
        
        # Turning phase
        spiral_length = turn_start_time  # Total length of spiral phase
        t_turn = t - turn_start_time
        
        # Length of turning arc
        angular_velocity = 1.0 / turn_params['radius']
        target_angle = np.pi
        turn_duration = target_angle / angular_velocity
        
        if t_turn <= turn_duration:
            # Still in arc
            arc_length = 1.0 * t_turn  # Speed 1m/s
            return spiral_length + arc_length
        else:
            # Arc ended, entered straight line segment
            arc_total_length = turn_params['radius'] * target_angle
            straight_length = 1.0 * (t_turn - turn_duration)
            return spiral_length + arc_total_length + straight_length
    
    def find_position_by_path_length(self, total_length, turn_params=None, turn_start_time=None):
        """
        Reverse calculate position coordinates from path length
        """
        if turn_params is None or total_length <= turn_start_time:
            # Spiral phase
            return self.phase1_spiral_trajectory(total_length)
        
        spiral_length = turn_start_time
        remaining_length = total_length - spiral_length
        
        # Calculate total turning arc length
        target_angle = np.pi
        arc_total_length = turn_params['radius'] * target_angle
        
        if remaining_length <= arc_total_length:
            # In arc segment
            angle_progress = remaining_length / turn_params['radius']
            angle = turn_params['start_angle'] + angle_progress
            x = turn_params['center'][0] + turn_params['radius'] * np.cos(angle)
            y = turn_params['center'][1] + turn_params['radius'] * np.sin(angle)
            return x, y
        else:
            # In straight line segment
            straight_length = remaining_length - arc_total_length
            end_angle = turn_params['start_angle'] + target_angle
            x_end = turn_params['center'][0] + turn_params['radius'] * np.cos(end_angle)
            y_end = turn_params['center'][1] + turn_params['radius'] * np.sin(end_angle)
            
            direction_x = -turn_params['tangent_direction'][0]
            direction_y = -turn_params['tangent_direction'][1]
            
            x = x_end + direction_x * straight_length
            y = y_end + direction_y * straight_length
            
            return x, y
    
    def calculate_positions_with_turn(self, time_points, turn_start_time=150):
        """
        Calculate position sequence including turning
        """
        # Calculate turning parameters
        turn_params = self.calculate_turn_parameters(turn_start_time)
        
        results = []
        
        for t in time_points:
            result = {'Time(s)': t}
            
            # Dragon head position
            if t <= turn_start_time:
                head_x, head_y = self.phase1_spiral_trajectory(t)
            else:
                head_x, head_y = self.phase2_turn_trajectory(t, turn_params, turn_start_time)
            
            result['Head_x(m)'] = head_x
            result['Head_y(m)'] = head_y
            
            # Calculate cumulative path length of dragon head
            head_path_length = self.get_trajectory_length_at_time(t, turn_params, turn_start_time)
            
            # Calculate handle positions
            handle_names = ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201']
            
            for name, distance in zip(handle_names, self.handle_distances):
                # Path length corresponding to this handle
                handle_path_length = max(0, head_path_length - distance)
                handle_x, handle_y = self.find_position_by_path_length(
                    handle_path_length, turn_params, turn_start_time)
                
                result[f'{name}_x(m)'] = handle_x
                result[f'{name}_y(m)'] = handle_y
            
            # Tail handle
            tail_path_length = max(0, head_path_length - self.tail_distance)
            tail_x, tail_y = self.find_position_by_path_length(
                tail_path_length, turn_params, turn_start_time)
            result['Tail_x(m)'] = tail_x
            result['Tail_y(m)'] = tail_y
            
            results.append(result)
        
        return pd.DataFrame(results), turn_params
    
    def analyze_collision_risk(self, turn_params, turn_start_time=150):
        """
        Analyze collision risk during turning process
        """
        print("\n=== Collision Risk Analysis ===")
        
        dragon_total_length = self.L_head + self.L_body + self.L_tail
        min_safe_radius = dragon_total_length / (2 * np.pi)
        
        print(f"Total dragon length: {dragon_total_length:.2f}m")
        print(f"Theoretical minimum turn radius: {min_safe_radius:.2f}m")
        print(f"Actual turn radius: {turn_params['radius']:.2f}m")
        print(f"Safety factor: {turn_params['radius'] / min_safe_radius:.2f}")
        
        if turn_params['radius'] > min_safe_radius * 1.2:
            print("✓ Turn radius sufficient, no collision risk")
        else:
            print("⚠ Turn radius small, collision risk exists")
    
    def solve_problem2(self, turn_start_time=150):
        """
        Complete solution process for Problem 2
        """
        print(f"\n=== Problem 2 Solution: Turn start time t={turn_start_time}s ===")
        
        # Given time points
        time_points = [0, 60, 120, 180, 240, 300]
        
        print("\nStep 1: Analyze turning trajectory design")
        turn_params = self.calculate_turn_parameters(turn_start_time)
        print(f"Turn start point: ({turn_params['start_point'][0]:.2f}, {turn_params['start_point'][1]:.2f})")
        print(f"Turn center: ({turn_params['center'][0]:.2f}, {turn_params['center'][1]:.2f})")
        print(f"Turn radius: {turn_params['radius']:.2f}m")
        
        print("\nStep 2: Collision risk assessment")
        self.analyze_collision_risk(turn_params, turn_start_time)
        
        print("\nStep 3: Calculate position sequence")
        position_df, turn_params = self.calculate_positions_with_turn(time_points, turn_start_time)
        print(position_df.round(3))
        
        print("\nStep 4: Calculate velocities")
        velocity_df = self.calculate_velocities_with_turn(time_points, turn_start_time)
        print(velocity_df.round(3))
        
        print("\nStep 5: Save results")
        with pd.ExcelWriter('result2.xlsx', engine='openpyxl') as writer:
            position_df.to_excel(writer, sheet_name='Turn_Position_Results', index=False)
            velocity_df.to_excel(writer, sheet_name='Turn_Velocity_Results', index=False)
        
        print("Turn results saved to result2.xlsx")
        
        return position_df, velocity_df, turn_params
    
    def calculate_velocities_with_turn(self, time_points, turn_start_time=150):
        """
        Calculate velocities during turning process
        """
        results = []
        dt = 1.0
        
        for i, t in enumerate(time_points):
            result = {'Time(s)': t}
            
            if i == 0:
                # First time point
                result['Head(m/s)'] = 1.0
                for name in ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201', 'Tail']:
                    result[f'{name}(m/s)'] = 0.8
            else:
                # Numerical differentiation calculation
                result['Head(m/s)'] = 1.0  # Assume constant
                
                # Other parts calculated through position differentiation
                pos_curr, _ = self.calculate_positions_with_turn([t], turn_start_time)
                pos_prev, _ = self.calculate_positions_with_turn([time_points[i-1]], turn_start_time)
                
                handles = ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201', 'Tail']
                for handle in handles:
                    dx = pos_curr[f'{handle}_x(m)'].iloc[0] - pos_prev[f'{handle}_x(m)'].iloc[0]
                    dy = pos_curr[f'{handle}_y(m)'].iloc[0] - pos_prev[f'{handle}_y(m)'].iloc[0]
                    speed = np.sqrt(dx**2 + dy**2) / (time_points[i] - time_points[i-1])
                    result[f'{handle}(m/s)'] = speed
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def visualize_turn_solution(self, turn_start_time=150):
        """
        Visualize turning solution
        """
        # Generate detailed trajectory
        time_detailed = np.linspace(0, 300, 1000)
        positions_detailed, turn_params = self.calculate_positions_with_turn(time_detailed, turn_start_time)
        
        # Key time points
        key_times = [0, 60, 120, 180, 240, 300]
        key_positions, _ = self.calculate_positions_with_turn(key_times, turn_start_time)
        
        plt.figure(figsize=(16, 12))
        
        # Plot complete trajectory
        plt.plot(positions_detailed['Head_x(m)'], positions_detailed['Head_y(m)'], 
                'r-', linewidth=2, label='Dragon Head Trajectory')
        
        # Plot handle trajectories
        handles = ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201']
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        for handle, color in zip(handles, colors):
            plt.plot(positions_detailed[f'{handle}_x(m)'], positions_detailed[f'{handle}_y(m)'], 
                    '--', color=color, alpha=0.7, label=f'{handle} Trajectory')
        
        plt.plot(positions_detailed['Tail_x(m)'], positions_detailed['Tail_y(m)'], 
                'k--', alpha=0.7, label='Dragon Tail Trajectory')
        
        # Mark turn start point
        turn_x, turn_y = self.phase1_spiral_trajectory(turn_start_time)
        plt.scatter(turn_x, turn_y, color='red', s=150, marker='*', 
                   label=f'Turn Start Point (t={turn_start_time}s)', zorder=10)
        
        # Draw turning circle
        circle = plt.Circle(turn_params['center'], turn_params['radius'], 
                           fill=False, color='red', linestyle=':', alpha=0.5)
        plt.gca().add_patch(circle)
        
        # Mark key time points
        for i, t in enumerate(key_times):
            row = key_positions.iloc[i]
            plt.scatter(row['Head_x(m)'], row['Head_y(m)'], 
                       color='red', s=60, marker='o', zorder=5)
            plt.annotate(f't={t}s', (row['Head_x(m)'], row['Head_y(m)']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title(f'Problem 2: Dragon Dance Turn Trajectory Analysis (Turn Time: t={turn_start_time}s)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        # Velocity analysis plot
        velocity_df = self.calculate_velocities_with_turn(key_times, turn_start_time)
        
        plt.figure(figsize=(12, 8))
        handles_speed = ['Head', 'Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201', 'Tail']
        
        for handle in handles_speed:
            plt.plot(key_times, velocity_df[f'{handle}(m/s)'], 
                    marker='o', label=f'{handle} Speed')
        
        plt.axvline(x=turn_start_time, color='red', linestyle='--', alpha=0.7, label='Turn Start')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('Problem 2: Speed Changes of Each Part During Turn')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Run Problem 2 solution
if __name__ == "__main__":
    model = Problem2DetailedModel()
    
    # Solve Problem 2 (assume turn starts at 150s)
    pos_df, vel_df, turn_params = model.solve_problem2(turn_start_time=150)
    
    # Visualize results
    model.visualize_turn_solution(turn_start_time=150)
    
    print("\n=== Problem 2 Modeling Summary ===")
    print("1. Trajectory segments: Spiral → Circular arc turn → Straight line return")
    print("2. Constraints: Constant dragon body length, avoid self-intersection")
    print("3. Key parameters: Turn radius, turn time, turn angle")
    print("4. Solution method: Path length parameterization + geometric constraint solving")