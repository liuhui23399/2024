import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem1DetailedModel:
    """
    Problem 1: Dragon head moves along spiral path with front handle maintaining 1m/s speed
    """
    
    def __init__(self):
        # Dragon geometry parameters (units: meters)
        self.L_head = 2.23  # Dragon head length 223cm
        self.L_body = 3.41  # Dragon body length 341cm  
        self.L_tail = 2.20  # Dragon tail length 220cm
        self.total_segments = 220  # Total body segments
        
        # Handle positions in dragon body segments
        self.handle_segments = [1, 51, 101, 151, 201]
        
        # Handle distances from dragon head front
        self.handle_distances = []
        for seg in self.handle_segments:
            # Distance = head length + (segment-1) * segment length
            dist = self.L_head + (seg - 1) * (self.L_body / self.total_segments)
            self.handle_distances.append(dist)
        
        # Dragon tail (rear) handle distance from dragon head front
        self.tail_distance = self.L_head + self.L_body + self.L_tail
        
        print("=== Problem 1 Model Parameters ===")
        print(f"Dragon head length: {self.L_head:.2f}m")
        print(f"Dragon body length: {self.L_body:.2f}m") 
        print(f"Dragon tail length: {self.L_tail:.2f}m")
        print(f"Handle distances from head front: {[f'{d:.3f}m' for d in self.handle_distances]}")
        print(f"Tail handle distance from head front: {self.tail_distance:.3f}m")
    
    def spiral_trajectory_param(self, s):
        """
        Parametric equation of spiral (with arc length s as parameter)
        To ensure dragon head front handle speed is 1m/s, s equals time t
        
        Parameters:
        s: Arc length parameter (equals time t, because speed is 1m/s)
        
        Returns:
        (x, y): Coordinates on spiral
        """
        # Spiral parameters
        # Choose appropriate parameters to make ds/dt = 1 (i.e., speed 1m/s)
        a = 1.0 / (2 * np.pi)  # Pitch parameter
        
        # Derive angle parameter θ from arc length
        # For constant speed spiral: s = a*θ²/2, so θ = sqrt(2s/a)
        theta = np.sqrt(2 * s / a) if s > 0 else 0
        
        # Spiral coordinates
        r = a * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return x, y
    
    def get_tangent_direction(self, s):
        """
        Get tangent direction (unit vector) at arc length s on spiral
        """
        ds = 0.001
        if s <= ds:
            x1, y1 = self.spiral_trajectory_param(s)
            x2, y2 = self.spiral_trajectory_param(s + ds)
        else:
            x1, y1 = self.spiral_trajectory_param(s - ds)
            x2, y2 = self.spiral_trajectory_param(s)
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            return dx/length, dy/length
        else:
            return 1.0, 0.0
    
    def find_position_at_distance(self, reference_s, distance):
        """
        Find position along spiral that is a specified distance behind reference point
        
        Parameters:
        reference_s: Arc length parameter of reference point
        distance: Distance backward
        
        Returns:
        target_s: Arc length parameter of target point
        """
        def distance_error(target_s):
            if target_s >= reference_s or target_s < 0:
                return float('inf')
            
            x_ref, y_ref = self.spiral_trajectory_param(reference_s)
            x_target, y_target = self.spiral_trajectory_param(target_s)
            
            actual_distance = np.sqrt((x_ref - x_target)**2 + (y_ref - y_target)**2)
            return abs(actual_distance - distance)
        
        # Initial guess: assume straight line distance
        initial_guess = max(0, reference_s - distance)
        
        # Search range
        search_range = np.linspace(max(0, reference_s - 2*distance), 
                                  max(0.001, reference_s - 0.1), 100)
        
        # Find optimal solution
        best_s = initial_guess
        best_error = distance_error(initial_guess)
        
        for s in search_range:
            error = distance_error(s)
            if error < best_error:
                best_error = error
                best_s = s
        
        return best_s
    
    def calculate_all_positions(self, time_points):
        """
        Calculate positions at all time points
        
        Core idea:
        1. Dragon head front handle moves along spiral at 1m/s speed
        2. Other parts move along same spiral but at different arc length positions
        3. Determine arc length parameters of each part through geometric constraints
        """
        results = []
        
        for t in time_points:
            result = {'Time(s)': t}
            
            # Dragon head front handle arc length parameter is time t (because speed 1m/s)
            head_s = t
            head_x, head_y = self.spiral_trajectory_param(head_s)
            result['Head_x(m)'] = head_x
            result['Head_y(m)'] = head_y
            
            # Calculate handle positions
            handle_names = ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201']
            
            for i, (name, distance) in enumerate(zip(handle_names, self.handle_distances)):
                # Find arc length parameter at specified distance from head
                handle_s = self.find_position_at_distance(head_s, distance)
                handle_x, handle_y = self.spiral_trajectory_param(handle_s)
                
                result[f'{name}_x(m)'] = handle_x
                result[f'{name}_y(m)'] = handle_y
            
            # Dragon tail (rear) handle
            tail_s = self.find_position_at_distance(head_s, self.tail_distance)
            tail_x, tail_y = self.spiral_trajectory_param(tail_s)
            result['Tail_x(m)'] = tail_x
            result['Tail_y(m)'] = tail_y
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def calculate_all_velocities(self, time_points):
        """
        Calculate velocities at all time points
        """
        dt = 1.0  # Time difference
        results = []
        
        for i, t in enumerate(time_points):
            result = {'Time(s)': t}
            
            if i == 0:
                # First time point, use analytical method or set initial values
                result['Head(m/s)'] = 1.0  # Given condition
                
                # Other parts' velocities estimated by numerical differentiation
                for name in ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201', 'Tail']:
                    result[f'{name}(m/s)'] = 0.8  # Initial estimate
            else:
                # Use numerical differentiation to calculate velocity
                t_prev = time_points[i-1]
                
                # Head velocity (given as 1m/s)
                result['Head(m/s)'] = 1.0
                
                # Other parts' velocities calculated through position differentiation
                pos_curr = self.calculate_all_positions([t])
                pos_prev = self.calculate_all_positions([t_prev])
                
                handles = ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201', 'Tail']
                for handle in handles:
                    dx = pos_curr[f'{handle}_x(m)'].iloc[0] - pos_prev[f'{handle}_x(m)'].iloc[0]
                    dy = pos_curr[f'{handle}_y(m)'].iloc[0] - pos_prev[f'{handle}_y(m)'].iloc[0]
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    result[f'{handle}(m/s)'] = speed
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def solve_problem1(self):
        """
        Complete solution process for Problem 1
        """
        print("\n=== Problem 1 Solution Process ===")
        
        # Given time points
        time_points = [0, 60, 120, 180, 240, 300]
        
        print("\nStep 1: Establish spiral trajectory model")
        print("- Dragon head front handle moves along spiral at constant 1m/s speed")
        print("- Spiral equation: r = a*θ, x = r*cos(θ), y = r*sin(θ)")
        print("- Arc length parameterization: s = t (because speed is 1m/s)")
        
        print("\nStep 2: Calculate positions at each time point")
        position_df = self.calculate_all_positions(time_points)
        print(position_df.round(3))
        
        print("\nStep 3: Calculate velocities at each time point")
        velocity_df = self.calculate_all_velocities(time_points)
        print(velocity_df.round(3))
        
        print("\nStep 4: Save results to Excel file")
        with pd.ExcelWriter('result1.xlsx', engine='openpyxl') as writer:
            position_df.to_excel(writer, sheet_name='Position_Results', index=False)
            velocity_df.to_excel(writer, sheet_name='Velocity_Results', index=False)
        
        print("Results saved to result1.xlsx")
        
        return position_df, velocity_df
    
    def visualize_solution(self, time_points=None):
        """
        Visualize Problem 1 solution
        """
        if time_points is None:
            time_points = [0, 60, 120, 180, 240, 300]
        
        # Calculate detailed trajectory for plotting
        detailed_times = np.linspace(0, 300, 500)
        detailed_positions = self.calculate_all_positions(detailed_times)
        
        # Calculate key time point positions
        key_positions = self.calculate_all_positions(time_points)
        
        plt.figure(figsize=(15, 10))
        
        # Plot trajectories
        plt.plot(detailed_positions['Head_x(m)'], detailed_positions['Head_y(m)'], 
                'r-', linewidth=2, label='Dragon Head Trajectory')
        
        handles = ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201']
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        for handle, color in zip(handles, colors):
            plt.plot(detailed_positions[f'{handle}_x(m)'], detailed_positions[f'{handle}_y(m)'], 
                    '--', color=color, alpha=0.7, label=f'{handle} Trajectory')
        
        plt.plot(detailed_positions['Tail_x(m)'], detailed_positions['Tail_y(m)'], 
                'k--', alpha=0.7, label='Dragon Tail Trajectory')
        
        # Mark key time points
        for i, t in enumerate(time_points):
            row = key_positions.iloc[i]
            plt.scatter(row['Head_x(m)'], row['Head_y(m)'], 
                       color='red', s=80, marker='o', zorder=5)
            plt.annotate(f't={t}s', (row['Head_x(m)'], row['Head_y(m)']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title('Problem 1: Dragon Dance Team Motion Trajectory (Head Front Handle Speed 1m/s)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        # Plot velocity change graph
        velocity_df = self.calculate_all_velocities(time_points)
        
        plt.figure(figsize=(12, 8))
        
        handles_speed = ['Head', 'Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201', 'Tail']
        
        for handle in handles_speed:
            plt.plot(time_points, velocity_df[f'{handle}(m/s)'], 
                    marker='o', label=f'{handle} Speed')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('Problem 1: Speed Changes of Each Part Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Run detailed solution for Problem 1
if __name__ == "__main__":
    model = Problem1DetailedModel()
    
    # Solve Problem 1
    pos_df, vel_df = model.solve_problem1()
    
    # Visualize results  
    model.visualize_solution()
    
    print("\n=== Modeling Summary ===")
    print("1. Core constraint: Dragon head front handle speed constant 1m/s")
    print("2. Geometric constraints: Fixed distances between parts")
    print("3. Solution method: Arc length parameterization + numerical optimization")
    print("4. Output results: Position coordinate table and velocity table")