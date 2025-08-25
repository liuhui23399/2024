import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
import math

# Set matplotlib to support Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem2CollisionDetection:
    """
    Problem 2: Dragon collision detection and terminal time calculation
    """
    
    def __init__(self):
        # Dragon geometry parameters (units: meters)
        self.L_head = 3.41  # Dragon head length 341cm
        self.L_body_segment = 2.20  # Each body/tail segment length 220cm
        self.total_segments = 223  # Total segments (1 head + 221 body + 1 tail)
        self.board_width = 0.30  # Board width 30cm
        
        # Spiral parameters from Problem 1
        self.spiral_pitch = 0.55  # Pitch 55cm
        self.head_speed = 1.0  # 1 m/s
        
        # Handle positions
        self.handle_segments = [1, 51, 101, 151, 201]
        self.handle_distances = []
        for seg in self.handle_segments:
            dist = self.L_head + (seg - 1) * self.L_body_segment
            self.handle_distances.append(dist)
        
        # Dragon tail distance
        self.tail_distance = self.L_head + (self.total_segments - 1) * self.L_body_segment
        
        print("=== Problem 2: Collision Detection Model ===")
        print(f"Dragon head length: {self.L_head:.2f}m")
        print(f"Body/tail segment length: {self.L_body_segment:.2f}m")
        print(f"Total dragon length: {self.tail_distance:.2f}m")
        print(f"Board width: {self.board_width:.2f}m")
        print(f"Spiral pitch: {self.spiral_pitch:.2f}m")
    
    def spiral_equation(self, theta):
        """
        Archimedean spiral equation r = a*θ
        """
        a = self.spiral_pitch / (2 * np.pi)
        r = a * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return r, x, y
    
    def arc_length_to_theta(self, s):
        """
        Convert arc length to angle parameter for spiral
        For Archimedean spiral: s ≈ (a*θ²)/2 for large θ
        """
        a = self.spiral_pitch / (2 * np.pi)
        # Approximation for large theta: theta ≈ sqrt(2s/a)
        theta = np.sqrt(2 * s / a) if s > 0 else 0
        return theta
    
    def get_dragon_positions(self, t):
        """
        Get positions of all dragon parts at time t
        """
        # Dragon head position
        head_arc_length = t * self.head_speed  # s = vt
        theta_head = self.arc_length_to_theta(head_arc_length)
        r_head, x_head, y_head = self.spiral_equation(theta_head)
        
        positions = [{'part': 'head', 'x': x_head, 'y': y_head, 'theta': theta_head}]
        
        # Handle positions
        for i, (seg, distance) in enumerate(zip(self.handle_segments, self.handle_distances)):
            handle_arc_length = head_arc_length - distance
            if handle_arc_length > 0:
                theta_handle = self.arc_length_to_theta(handle_arc_length)
                r_handle, x_handle, y_handle = self.spiral_equation(theta_handle)
                positions.append({
                    'part': f'body_seg{seg}',
                    'x': x_handle,
                    'y': y_handle,
                    'theta': theta_handle
                })
        
        # Dragon tail position
        tail_arc_length = head_arc_length - self.tail_distance
        if tail_arc_length > 0:
            theta_tail = self.arc_length_to_theta(tail_arc_length)
            r_tail, x_tail, y_tail = self.spiral_equation(theta_tail)
            positions.append({
                'part': 'tail',
                'x': x_tail,
                'y': y_tail,
                'theta': theta_tail
            })
        
        return positions
    
    def check_collision_at_time(self, t, min_distance_threshold=None):
        """
        Check if collision occurs at time t
        """
        if min_distance_threshold is None:
            min_distance_threshold = self.board_width
        
        positions = self.get_dragon_positions(t)
        
        # Check minimum distance between non-adjacent parts
        min_distance = float('inf')
        collision_pair = None
        
        for i in range(len(positions)):
            for j in range(i + 2, len(positions)):  # Skip adjacent parts
                x1, y1 = positions[i]['x'], positions[i]['y']
                x2, y2 = positions[j]['x'], positions[j]['y']
                
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    collision_pair = (positions[i]['part'], positions[j]['part'])
        
        collision = min_distance < min_distance_threshold
        
        return collision, min_distance, collision_pair, positions
    
    def find_collision_time(self):
        """
        Find the time when collision first occurs using binary search
        """
        print("\n=== Finding Collision Time ===")
        
        # Start with a more reasonable search range
        # Dragon needs time to spiral inward enough for collision
        t_min = 100.0  # Start from t=100s to allow spiral development
        t_max = 500.0  # More focused upper bound
        
        # First, check if collision occurs in reasonable range
        max_check_time = 1000.0
        collision_found = False
        
        # Sample several points to find collision
        test_times = np.linspace(t_min, max_check_time, 50)
        min_distances = []
        
        for t in test_times:
            collision, min_dist, pair, positions = self.check_collision_at_time(t)
            min_distances.append(min_dist)
            
            if collision:
                collision_found = True
                t_max = t
                print(f"Collision detected at t={t:.1f}s, min_distance={min_dist:.3f}m")
                break
            elif min_dist < 0.5:  # Getting close to collision
                print(f"Close approach at t={t:.1f}s, min_distance={min_dist:.3f}m")
        
        if not collision_found:
            # Check if the dragon is spiraling tight enough
            print(f"Minimum distance found: {min(min_distances):.3f}m")
            
            # Try with tighter collision threshold
            tight_threshold = min(min_distances) * 0.8
            print(f"Trying with tighter threshold: {tight_threshold:.3f}m")
            
            for t in test_times:
                collision, min_dist, pair, positions = self.check_collision_at_time(t, tight_threshold)
                if collision:
                    collision_found = True
                    t_max = t
                    print(f"Collision with tight threshold at t={t:.1f}s, min_distance={min_dist:.3f}m")
                    break
        
        if not collision_found:
            print("Dragon may not collide with current spiral parameters")
            # Return best estimate based on minimum distance
            min_dist_idx = np.argmin(min_distances)
            best_time = test_times[min_dist_idx]
            print(f"Returning closest approach time: {best_time:.1f}s, distance: {min_distances[min_dist_idx]:.3f}m")
            
            # Get positions at this time
            _, _, _, final_positions = self.check_collision_at_time(best_time)
            return best_time, final_positions, ("closest_approach", "estimated")
        
        # Binary search for exact collision time
        tolerance = 0.1  # 0.1 second precision
        max_iterations = 30
        
        for iteration in range(max_iterations):
            t_mid = (t_min + t_max) / 2
            collision, min_dist, pair, positions = self.check_collision_at_time(t_mid, tight_threshold if 'tight_threshold' in locals() else None)
            
            print(f"Iteration {iteration+1}: t={t_mid:.3f}s, collision={collision}, min_dist={min_dist:.3f}m")
            
            if collision:
                t_max = t_mid
            else:
                t_min = t_mid
            
            if t_max - t_min < tolerance:
                break
        
        # Final collision time
        collision_time = t_max
        final_collision, final_min_dist, final_pair, final_positions = self.check_collision_at_time(
            collision_time, tight_threshold if 'tight_threshold' in locals() else None
        )
        
        print(f"\n=== Collision Analysis Complete ===")
        print(f"Collision time: {collision_time:.3f}s")
        print(f"Minimum distance: {final_min_dist:.3f}m")
        print(f"Collision between: {final_pair}")
        
        return collision_time, final_positions, final_pair
    
    def calculate_velocities_at_time(self, t, dt=0.01):
        """
        Calculate velocities of all dragon parts at time t
        """
        positions_t1 = self.get_dragon_positions(t - dt/2)
        positions_t2 = self.get_dragon_positions(t + dt/2)
        
        velocities = []
        
        # Match positions by part name
        part_map = {pos['part']: pos for pos in positions_t1}
        
        for pos2 in positions_t2:
            part_name = pos2['part']
            if part_name in part_map:
                pos1 = part_map[part_name]
                
                dx = pos2['x'] - pos1['x']
                dy = pos2['y'] - pos1['y']
                speed = np.sqrt(dx**2 + dy**2) / dt
                
                velocities.append({
                    'part': part_name,
                    'speed': speed,
                    'vx': dx / dt,
                    'vy': dy / dt
                })
        
        return velocities
    
    def solve_problem2(self):
        """
        Complete solution for Problem 2
        """
        print("=== Problem 2: Finding Terminal Time ===")
        print("Objective: Determine when dragon collision first occurs")
        
        # Find collision time
        collision_time, final_positions, collision_pair = self.find_collision_time()
        
        if collision_time is None:
            print("No collision detected in reasonable time range")
            return None
        
        # Calculate velocities at collision time
        final_velocities = self.calculate_velocities_at_time(collision_time)
        
        # Prepare results
        results = []
        
        # Map positions and velocities
        pos_map = {pos['part']: pos for pos in final_positions}
        vel_map = {vel['part']: vel for vel in final_velocities}
        
        # Standard output format
        part_names = ['head', 'body_seg1', 'body_seg51', 'body_seg101', 'body_seg151', 'body_seg201', 'tail']
        display_names = ['龙头', '第1节龙身', '第51节龙身', '第101节龙身', '第151节龙身', '第201节龙身', '龙尾（后）']
        
        for part, display in zip(part_names, display_names):
            if part in pos_map and part in vel_map:
                pos = pos_map[part]
                vel = vel_map[part]
                results.append({
                    'Part': display,
                    'Time(s)': collision_time,
                    'X(m)': pos['x'],
                    'Y(m)': pos['y'],
                    'Speed(m/s)': vel['speed']
                })
        
        results_df = pd.DataFrame(results)
        
        print(f"\n=== Final Results ===")
        print(f"Terminal time: {collision_time:.3f}s")
        print(f"Collision between: {collision_pair}")
        print("\nPositions and velocities at terminal time:")
        print(results_df.round(6))
        
        # Save to CSV
        results_df.to_csv('result2.csv', index=False)
        print(f"\nResults saved to result2.csv")
        
        return collision_time, results_df, collision_pair
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
    
    def solve_problem2(self):
        """
        Complete solution for Problem 2
        """
        print("=== Problem 2: Finding Terminal Time ===")
        print("Objective: Determine when dragon collision first occurs")
        
        # Find collision time
        collision_time, final_positions, collision_pair = self.find_collision_time()
        
        if collision_time is None:
            print("No collision detected in reasonable time range")
            return None
        
        # Calculate velocities at collision time
        final_velocities = self.calculate_velocities_at_time(collision_time)
        
        # Prepare results
        results = []
        
        # Map positions and velocities
        pos_map = {pos['part']: pos for pos in final_positions}
        vel_map = {vel['part']: vel for vel in final_velocities}
        
        # Standard output format
        part_names = ['head', 'body_seg1', 'body_seg51', 'body_seg101', 'body_seg151', 'body_seg201', 'tail']
        display_names = ['龙头', '第1节龙身', '第51节龙身', '第101节龙身', '第151节龙身', '第201节龙身', '龙尾（后）']
        
        for part, display in zip(part_names, display_names):
            if part in pos_map and part in vel_map:
                pos = pos_map[part]
                vel = vel_map[part]
                results.append({
                    'Part': display,
                    'Time(s)': collision_time,
                    'X(m)': pos['x'],
                    'Y(m)': pos['y'],
                    'Speed(m/s)': vel['speed']
                })
        
        results_df = pd.DataFrame(results)
        
        print(f"\n=== Final Results ===")
        print(f"Terminal time: {collision_time:.3f}s")
        print(f"Collision between: {collision_pair}")
        print("\nPositions and velocities at terminal time:")
        print(results_df.round(6))
        
        # Save to CSV
        results_df.to_csv('result2.csv', index=False)
        print(f"\nResults saved to result2.csv")
        
        return collision_time, results_df, collision_pair
    
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
    model = Problem2CollisionDetection()
    
    # Solve Problem 2
    result = model.solve_problem2()
    
    if result and result[0] is not None:
        collision_time, results_df, collision_pair = result
        print(f"\n=== Problem 2 建模总结 ===")
        print(f"1. 碰撞检测方法：基于最小距离阈值判断")
        print(f"2. 关键约束：板凳宽度30cm，相邻板凳不能重叠")
        print(f"3. 求解方法：二分搜索找到首次碰撞时间")
        print(f"4. 终止时间：{collision_time:.3f}秒")
        print(f"5. 碰撞部位：{collision_pair}")
    else:
        print("\n未检测到明确碰撞，可能需要调整模型参数")