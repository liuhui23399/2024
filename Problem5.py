import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import math

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem5SpeedOptimization:
    """
    Problem 5: 龙头最大行进速度优化分析
    确定龙头的最大行进速度，使得舞蹈队各把手的速度均不超过2 m/s
    """
    
    def __init__(self):
        # 龙的几何参数（单位：米）
        self.L_head = 2.23  # 龙头长度
        self.L_body = 3.41  # 龙身长度  
        self.L_tail = 2.20  # 龙尾长度
        self.total_segments = 220  # 总体节数
        
        # 路径参数（来自问题4）
        self.spiral_pitch = 1.7  # 螺距 1.7m
        self.turnaround_radius = 4.5  # 调头空间半径 4.5m
        
        # 速度约束
        self.max_handle_speed = 2.0  # 各把手最大允许速度 2m/s
        
        # 龙身各把手位置
        self.handle_segments = [1, 51, 101, 151, 201]
        self.handle_distances = []
        for seg in self.handle_segments:
            dist = self.L_head + (seg - 1) * (self.L_body / self.total_segments)
            self.handle_distances.append(dist)
        
        # 龙尾距离
        self.tail_distance = self.L_head + self.L_body + self.L_tail
        
        print("=== Problem 5: Speed Optimization Model ===")
        print(f"Spiral pitch: {self.spiral_pitch}m")
        print(f"Turnaround radius: {self.turnaround_radius}m")
        print(f"Max handle speed constraint: {self.max_handle_speed}m/s")
        print(f"Handle distances: {[f'{d:.3f}m' for d in self.handle_distances]}")
        print(f"Tail distance: {self.tail_distance:.3f}m")
    
    def spiral_equation(self, theta, is_outward=False):
        """
        Archimedean spiral equation
        """
        a = self.spiral_pitch / (2 * np.pi)
        r = a * theta
        
        if is_outward:
            x = -r * np.cos(theta)
            y = -r * np.sin(theta)
        else:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        
        return r, x, y
    
    def calculate_curvature(self, theta):
        """
        Calculate curvature of Archimedean spiral at angle theta
        For r = a*θ: κ = (r² + 2a²) / (r² + a²)^(3/2)
        """
        if theta <= 0:
            return float('inf')  # Infinite curvature at origin
        
        a = self.spiral_pitch / (2 * np.pi)
        r = a * theta
        
        numerator = r**2 + 2 * a**2
        denominator = (r**2 + a**2)**(3/2)
        
        curvature = numerator / denominator if denominator > 0 else 0
        return curvature
    
    def calculate_speed_ratios(self, theta_head, head_speed):
        """
        Calculate speed ratios for all dragon parts using kinematic analysis
        """
        speeds = {'Head': head_speed}
        
        # Calculate curvature at head position
        curvature = self.calculate_curvature(theta_head)
        
        # For each handle, calculate its position and speed
        a = self.spiral_pitch / (2 * np.pi)
        
        handle_names = ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201']
        
        for i, (name, distance) in enumerate(zip(handle_names, self.handle_distances)):
            # Approximate the angle for this handle (based on arc length)
            head_arc_length = (a * theta_head**2) / 2
            handle_arc_length = max(0, head_arc_length - distance)
            
            if handle_arc_length > 0:
                theta_handle = np.sqrt(2 * handle_arc_length / a)
                
                # Calculate speed amplification due to curvature
                r_head = a * theta_head
                r_handle = a * theta_handle
                
                # Speed ratio based on geometric constraints
                radius_ratio = r_handle / r_head if r_head > 0 else 0
                curvature_handle = self.calculate_curvature(theta_handle)
                
                # Combined speed factor: base ratio + curvature amplification
                curvature_amplification = 1 + distance * curvature_handle / 10
                speed_ratio = radius_ratio * curvature_amplification
                
                speeds[name] = head_speed * speed_ratio
            else:
                speeds[name] = 0.0
        
        # Dragon tail
        head_arc_length = (a * theta_head**2) / 2
        tail_arc_length = max(0, head_arc_length - self.tail_distance)
        
        if tail_arc_length > 0:
            theta_tail = np.sqrt(2 * tail_arc_length / a)
            r_tail = a * theta_tail
            r_head = a * theta_head
            
            radius_ratio = r_tail / r_head if r_head > 0 else 0
            curvature_tail = self.calculate_curvature(theta_tail)
            curvature_amplification = 1 + self.tail_distance * curvature_tail / 10
            
            speed_ratio = radius_ratio * curvature_amplification
            speeds['Tail'] = head_speed * speed_ratio
        else:
            speeds['Tail'] = 0.0
        
        return speeds
    
    def find_max_speed_at_position(self, theta):
        """
        Find maximum allowable head speed at given spiral position
        """
        def check_speed_constraint(head_speed):
            speeds = self.calculate_speed_ratios(theta, head_speed)
            max_speed = max(speeds.values())
            return max_speed - self.max_handle_speed
        
        # Binary search for maximum allowable head speed
        low_speed = 0.01
        high_speed = 3.0
        tolerance = 1e-6
        
        # Check if high speed is feasible
        if check_speed_constraint(high_speed) <= 0:
            return high_speed
        
        # Binary search
        for _ in range(50):
            mid_speed = (low_speed + high_speed) / 2
            constraint_violation = check_speed_constraint(mid_speed)
            
            if abs(constraint_violation) < tolerance:
                break
            elif constraint_violation > 0:
                high_speed = mid_speed
            else:
                low_speed = mid_speed
        
        return (low_speed + high_speed) / 2
    
    def analyze_spiral_speed_constraints(self):
        """
        Analyze speed constraints along the spiral path
        """
        print("\n=== Analyzing Speed Constraints Along Spiral ===")
        
        # Calculate boundary angle
        a = self.spiral_pitch / (2 * np.pi)
        theta_boundary = self.turnaround_radius / a
        
        print(f"Spiral boundary angle: {theta_boundary:.3f} radians ({theta_boundary*180/np.pi:.1f}°)")
        
        # Sample points along spiral
        theta_points = np.linspace(0.5, theta_boundary, 25)  # Avoid θ=0 singularity
        
        results = []
        for theta in theta_points:
            r, x, y = self.spiral_equation(theta)
            curvature = self.calculate_curvature(theta)
            max_head_speed = self.find_max_speed_at_position(theta)
            
            # Test speeds at this position
            test_speeds = self.calculate_speed_ratios(theta, max_head_speed)
            max_actual_speed = max(test_speeds.values())
            
            results.append({
                'theta': theta,
                'radius': r,
                'x': x,
                'y': y,
                'curvature': curvature,
                'max_head_speed': max_head_speed,
                'max_actual_speed': max_actual_speed,
                'limiting_part': max(test_speeds.keys(), key=lambda k: test_speeds[k])
            })
            
            if len(results) % 5 == 0:
                print(f"θ={theta:.3f}, r={r:.3f}m, max_head_speed={max_head_speed:.4f}m/s")
        
        return pd.DataFrame(results)
    
    def analyze_turnaround_speed_constraints(self):
        """
        Analyze speed constraints during turnaround phase
        """
        print("\n=== Analyzing Turnaround Speed Constraints ===")
        
        # Estimate turnaround path curvature
        # Based on Problem 4 design: two circular arcs with radii in 2:1 ratio
        
        # Boundary points distance
        a = self.spiral_pitch / (2 * np.pi)
        theta_boundary = self.turnaround_radius / a
        _, x_in, y_in = self.spiral_equation(theta_boundary, is_outward=False)
        _, x_out, y_out = self.spiral_equation(theta_boundary, is_outward=True)
        
        distance_between = np.sqrt((x_out - x_in)**2 + (y_out - y_in)**2)
        
        # Arc radii (from Problem 4 design)
        r2 = distance_between / 8  # Smaller radius
        r1 = 2 * r2  # Larger radius (2:1 ratio)
        
        # Maximum curvature in turnaround
        max_curvature = 1 / r2  # Smaller radius has higher curvature
        
        print(f"Turnaround arc radii: {r1:.3f}m, {r2:.3f}m")
        print(f"Maximum curvature: {max_curvature:.6f} 1/m")
        
        # Speed constraint based on maximum curvature
        # For the part with maximum distance (tail), speed amplification is highest
        max_distance = self.tail_distance
        
        # Simplified speed amplification model for circular arc
        # Speed ratio ≈ 1 + distance * curvature / 2
        max_amplification = 1 + max_distance * max_curvature / 2
        
        turnaround_max_speed = self.max_handle_speed / max_amplification
        
        print(f"Maximum speed amplification factor: {max_amplification:.3f}")
        print(f"Turnaround maximum head speed: {turnaround_max_speed:.4f}m/s")
        
        return turnaround_max_speed, max_amplification
    
    def find_global_maximum_speed(self):
        """
        Find global maximum allowable head speed
        """
        print("\n=== Finding Global Maximum Speed ===")
        
        # Analyze spiral constraints
        spiral_analysis = self.analyze_spiral_speed_constraints()
        spiral_min_speed = spiral_analysis['max_head_speed'].min()
        spiral_bottleneck = spiral_analysis.loc[spiral_analysis['max_head_speed'].idxmin()]
        
        print("\nSpiral phase bottleneck:")
        print(f"  Position: θ={spiral_bottleneck['theta']:.3f}, r={spiral_bottleneck['radius']:.3f}m")
        print(f"  Coordinates: ({spiral_bottleneck['x']:.3f}, {spiral_bottleneck['y']:.3f})")
        print(f"  Curvature: {spiral_bottleneck['curvature']:.6f} 1/m")
        print(f"  Max head speed: {spiral_min_speed:.4f}m/s")
        print(f"  Limiting part: {spiral_bottleneck['limiting_part']}")
        
        # Analyze turnaround constraints
        turnaround_max_speed, turnaround_amplification = self.analyze_turnaround_speed_constraints()
        
        # Global maximum is the minimum of all constraints
        global_max_speed = min(spiral_min_speed, turnaround_max_speed)
        
        print("\n=== Final Results ===")
        print(f"Spiral phase max speed: {spiral_min_speed:.4f}m/s")
        print(f"Turnaround phase max speed: {turnaround_max_speed:.4f}m/s")
        print(f"Global maximum head speed: {global_max_speed:.4f}m/s")
        
        # Verify final result
        print("\n=== Verification ===")
        bottleneck_speeds = self.calculate_speed_ratios(
            spiral_bottleneck['theta'], global_max_speed
        )
        
        print("Speeds at global maximum:")
        for part, speed in bottleneck_speeds.items():
            constraint_met = "✓" if speed <= self.max_handle_speed else "✗"
            print(f"  {part}: {speed:.4f}m/s {constraint_met}")
        
        return global_max_speed, spiral_analysis, spiral_bottleneck
    
    def solve_problem5(self):
        """
        Complete solution for Problem 5
        """
        print("=== Problem 5: Maximum Head Speed Optimization ===")
        print("Objective: Find maximum head speed such that all handle speeds ≤ 2 m/s")
        
        # Find global maximum speed
        global_max_speed, spiral_analysis, bottleneck_info = self.find_global_maximum_speed()
        
        # Save detailed analysis (改为CSV格式)
        spiral_analysis.to_csv('result5.csv', index=False)
        print("\nDetailed analysis saved to result5.csv")
        
        # 可视化分析结果
        self.visualize_speed_analysis(spiral_analysis, global_max_speed)
        
        # Create summary report
        summary = {
            'Maximum Head Speed (m/s)': global_max_speed,
            'Bottleneck Location': f"θ={bottleneck_info['theta']:.3f}",
            'Bottleneck Curvature (1/m)': bottleneck_info['curvature'],
            'Limiting Part': bottleneck_info['limiting_part']
        }
        
        print("\n=== Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return global_max_speed, spiral_analysis
    
    def visualize_speed_analysis(self, spiral_analysis, global_max_speed):
        """
        可视化速度分析结果 - 4个子图完整分析
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 螺线路径上的最大允许速度分布
        ax1.plot(spiral_analysis['theta'], spiral_analysis['max_head_speed'], 'b-o', linewidth=2, markersize=4)
        ax1.axhline(y=global_max_speed, color='r', linestyle='--', linewidth=2,
                   label=f'全局最大速度 = {global_max_speed:.4f} m/s')
        ax1.set_xlabel('角度 θ (弧度)', fontsize=12)
        ax1.set_ylabel('最大允许龙头速度 (m/s)', fontsize=12)
        ax1.set_title('沿螺线的最大允许速度分布', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标记瓶颈点
        min_speed_idx = spiral_analysis['max_head_speed'].idxmin()
        bottleneck = spiral_analysis.iloc[min_speed_idx]
        ax1.scatter(bottleneck['theta'], bottleneck['max_head_speed'], 
                   color='red', s=100, marker='*', zorder=5)
        
        # 2. 曲率分布
        ax2.plot(spiral_analysis['theta'], spiral_analysis['curvature'], 'g-o', linewidth=2, markersize=4)
        ax2.set_xlabel('角度 θ (弧度)', fontsize=12)
        ax2.set_ylabel('曲率 (1/m)', fontsize=12)
        ax2.set_title('螺线曲率分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 标记最大曲率点
        max_curvature_idx = spiral_analysis['curvature'].idxmax()
        max_curvature_point = spiral_analysis.iloc[max_curvature_idx]
        ax2.scatter(max_curvature_point['theta'], max_curvature_point['curvature'], 
                   color='red', s=100, marker='*', zorder=5, label='最大曲率点')
        ax2.legend()
        
        # 3. 螺线轨迹和速度标注
        # 计算螺线轨迹坐标
        x_coords = []
        y_coords = []
        for _, row in spiral_analysis.iterrows():
            r, x, y = self.spiral_equation(row['theta'], is_outward=False)
            x_coords.append(x)
            y_coords.append(y)
        
        ax3.plot(x_coords, y_coords, 'b-', linewidth=2, label='螺线轨迹')
        
        # 标注速度瓶颈点
        bottleneck_r, bottleneck_x, bottleneck_y = self.spiral_equation(bottleneck['theta'], is_outward=False)
        ax3.scatter(bottleneck_x, bottleneck_y, color='red', s=200, marker='*', 
                   label=f'速度瓶颈点: {bottleneck["max_head_speed"]:.4f} m/s')
        
        # 调头空间边界
        circle = plt.Circle((0, 0), self.turnaround_radius, fill=False, color='red', 
                           linestyle='--', linewidth=2, label='调头空间边界')
        ax3.add_patch(circle)
        
        ax3.set_xlabel('X坐标 (m)', fontsize=12)
        ax3.set_ylabel('Y坐标 (m)', fontsize=12)
        ax3.set_title('螺线轨迹与速度分析', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # 4. 各把手速度对比（在瓶颈点）
        bottleneck_speeds = self.calculate_speed_ratios(bottleneck['theta'], bottleneck['max_head_speed'])
        
        handles = list(bottleneck_speeds.keys())
        speeds = list(bottleneck_speeds.values())
        
        # 设置颜色：龙头为红色，其他为蓝色
        colors = ['red' if handle == 'Head' else 'blue' for handle in handles]
        bars = ax4.bar(range(len(handles)), speeds, color=colors, alpha=0.7)
        
        # 添加速度上限线
        ax4.axhline(y=self.max_handle_speed, color='red', linestyle='--', linewidth=2,
                   label=f'速度上限 = {self.max_handle_speed} m/s')
        
        ax4.set_xlabel('把手位置', fontsize=12)
        ax4.set_ylabel('速度 (m/s)', fontsize=12)
        ax4.set_title(f'瓶颈点各把手速度分布\n(龙头速度 = {bottleneck["max_head_speed"]:.4f} m/s)', 
                     fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(handles)))
        ax4.set_xticklabels(handles, rotation=45, fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 在每个柱子上添加数值标签
        for i, (bar, speed) in enumerate(zip(bars, speeds)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{speed:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('problem5_speed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("完整可视化图表已保存到 problem5_speed_analysis.png")

# Run Problem 5 solution
if __name__ == "__main__":
    solver = Problem5SpeedOptimization()
    max_speed, analysis = solver.solve_problem5()
    
    print(f"\n=== Problem 5 最终答案 ===")
    print(f"龙头的最大行进速度: {max_speed:.4f} m/s")
    
    print(f"\n=== 建模总结 ===")
    print(f"1. 约束条件：所有把手速度不超过2 m/s")
    print(f"2. 关键因素：螺线曲率导致的速度放大效应")
    print(f"3. 瓶颈位置：螺线中曲率较大的区域")
    print(f"4. 求解方法：基于曲率分析的速度优化")
    print(f"5. 实际意义：为龙舞表演提供安全速度指导")
