import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar
import pandas as pd

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem3Solution:
    """
    问题3: 龙头舞蹈队调头空间分析
    确定最小螺距，使得龙头前把手能够沿着相应的螺线盘入到调头空间的边界
    """
    
    def __init__(self):
        # 龙的几何参数（单位：米）
        self.L_head = 2.23  # 龙头长度 223cm
        self.L_body = 3.41  # 龙身长度 341cm  
        self.L_tail = 2.20  # 龙尾长度 220cm
        self.total_segments = 220  # 总体节数
        
        # 调头空间参数
        self.turnaround_diameter = 9.0  # 调头空间直径 9m
        self.turnaround_radius = self.turnaround_diameter / 2  # 调头空间半径 4.5m
        
        # 龙头前把手速度
        self.head_speed = 1.0  # 1 m/s
        
        print("=== 问题3 模型参数 ===")
        print(f"龙头长度: {self.L_head:.2f}m")
        print(f"龙身长度: {self.L_body:.2f}m") 
        print(f"龙尾长度: {self.L_tail:.2f}m")
        print(f"调头空间直径: {self.turnaround_diameter:.1f}m")
        print(f"调头空间半径: {self.turnaround_radius:.1f}m")
    
    def spiral_equation(self, theta, pitch):
        """
        螺线方程
        参数:
        theta: 角度参数
        pitch: 螺距（相邻两圈之间的径向距离）
        
        返回:
        (r, x, y): 极径和直角坐标
        """
        # 螺距参数 a = pitch / (2π)
        a = pitch / (2 * np.pi)
        r = a * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return r, x, y
    
    def find_entry_angle(self, pitch):
        """
        找到龙头前把手刚好到达调头空间边界时的角度
        """
        def radius_error(theta):
            r, _, _ = self.spiral_equation(theta, pitch)
            return abs(r - self.turnaround_radius)
        
        # 寻找使得r = turnaround_radius的theta值
        # 初始猜测：theta应该在几圈到十几圈之间
        theta_range = np.linspace(2*np.pi, 20*np.pi, 1000)
        
        best_theta = 2*np.pi
        min_error = float('inf')
        
        for theta in theta_range:
            error = radius_error(theta)
            if error < min_error:
                min_error = error
                best_theta = theta
        
        return best_theta, min_error
    
    def calculate_dragon_positions(self, theta_head, pitch):
        """
        计算龙头到达指定角度时，整条龙各部分的位置
        """
        # 龙头前把手位置
        r_head, x_head, y_head = self.spiral_equation(theta_head, pitch)
        
        # 计算龙身各把手位置（需要考虑沿螺线的距离约束）
        handle_segments = [1, 51, 101, 151, 201]  # 第1, 51, 101, 151, 201节龙身把手
        handle_distances = []
        
        for seg in handle_segments:
            # 距离龙头前把手的距离
            dist = self.L_head + (seg - 1) * (self.L_body / self.total_segments)
            handle_distances.append(dist)
        
        # 龙尾后把手距离
        tail_distance = self.L_head + self.L_body + self.L_tail
        
        # 找到各把手对应的角度（需要数值求解）
        positions = {'head': (theta_head, x_head, y_head, r_head)}
        
        for i, (seg, distance) in enumerate(zip(handle_segments, handle_distances)):
            theta_handle = self.find_position_on_spiral(theta_head, distance, pitch)
            r_handle, x_handle, y_handle = self.spiral_equation(theta_handle, pitch)
            positions[f'body_seg{seg}'] = (theta_handle, x_handle, y_handle, r_handle)
        
        # 龙尾把手
        theta_tail = self.find_position_on_spiral(theta_head, tail_distance, pitch)
        r_tail, x_tail, y_tail = self.spiral_equation(theta_tail, pitch)
        positions['tail'] = (theta_tail, x_tail, y_tail, r_tail)
        
        return positions
    
    def find_position_on_spiral(self, theta_head, distance, pitch):
        """
        在螺线上找到距离龙头指定距离的点对应的角度
        """
        def distance_error(theta_target):
            if theta_target >= theta_head or theta_target < 0:
                return float('inf')
            
            # 计算两点间的实际距离（沿螺线）
            # 简化处理：使用直线距离近似
            r_head, x_head, y_head = self.spiral_equation(theta_head, pitch)
            r_target, x_target, y_target = self.spiral_equation(theta_target, pitch)
            
            actual_distance = np.sqrt((x_head - x_target)**2 + (y_head - y_target)**2)
            return abs(actual_distance - distance)
        
        # 搜索范围
        search_range = np.linspace(max(0.1, theta_head - 4*np.pi), 
                                  max(0.1, theta_head - 0.1), 100)
        
        best_theta = theta_head - np.pi
        best_error = float('inf')
        
        for theta in search_range:
            error = distance_error(theta)
            if error < best_error:
                best_error = error
                best_theta = theta
        
        return best_theta
    
    def check_all_parts_in_boundary(self, pitch):
        """
        检查给定螺距下，龙头到达边界时整条龙是否都在调头空间内
        """
        # 找到龙头到达边界时的角度
        theta_head, _ = self.find_entry_angle(pitch)
        
        # 计算所有部分的位置
        positions = self.calculate_dragon_positions(theta_head, pitch)
        
        # 检查是否所有部分都在调头空间内
        all_inside = True
        max_radius = 0
        
        for part, (theta, x, y, r) in positions.items():
            if r > max_radius:
                max_radius = r
            if r > self.turnaround_radius:
                all_inside = False
        
        return all_inside, max_radius, positions
    
    def find_minimum_pitch(self):
        """
        寻找最小螺距
        """
        print("\n=== 寻找最小螺距 ===")
        
        # 螺距搜索范围：从很小的值开始搜索
        pitch_range = np.linspace(0.1, 2.0, 200)
        
        valid_pitches = []
        results = []
        
        for pitch in pitch_range:
            all_inside, max_radius, positions = self.check_all_parts_in_boundary(pitch)
            
            result = {
                'pitch': pitch,
                'all_inside': all_inside,
                'max_radius': max_radius,
                'margin': self.turnaround_radius - max_radius
            }
            results.append(result)
            
            if all_inside:
                valid_pitches.append(pitch)
                print(f"螺距 {pitch:.3f}m: 可行, 最大半径 {max_radius:.3f}m, 余量 {self.turnaround_radius - max_radius:.3f}m")
        
        if valid_pitches:
            min_pitch = min(valid_pitches)
            print(f"\n找到最小螺距: {min_pitch:.3f}m")
            
            # 验证最小螺距的详细结果
            all_inside, max_radius, positions = self.check_all_parts_in_boundary(min_pitch)
            print(f"验证结果:")
            print(f"  - 所有部分都在边界内: {all_inside}")
            print(f"  - 最大半径: {max_radius:.3f}m")
            print(f"  - 安全余量: {self.turnaround_radius - max_radius:.3f}m")
            
            return min_pitch, results, positions
        else:
            print("在搜索范围内未找到可行的螺距")
            return None, results, None
    
    def visualize_solution(self, min_pitch, positions):
        """
        可视化解决方案
        """
        if min_pitch is None or positions is None:
            print("无法可视化：未找到可行解")
            return
        
        plt.figure(figsize=(12, 10))
        
        # 绘制调头空间边界
        circle = plt.Circle((0, 0), self.turnaround_radius, 
                           fill=False, color='red', linewidth=3, 
                           label=f'调头空间边界 (R={self.turnaround_radius}m)')
        plt.gca().add_patch(circle)
        
        # 绘制螺线轨迹
        theta_max = max([pos[0] for pos in positions.values()]) + np.pi
        theta_range = np.linspace(0, theta_max, 1000)
        
        spiral_x = []
        spiral_y = []
        for theta in theta_range:
            _, x, y = self.spiral_equation(theta, min_pitch)
            spiral_x.append(x)
            spiral_y.append(y)
        
        plt.plot(spiral_x, spiral_y, 'b--', alpha=0.5, linewidth=1, 
                label=f'螺线轨迹 (螺距={min_pitch:.3f}m)')
        
        # 标记龙的各部分位置
        colors = {'head': 'red', 'tail': 'purple'}
        markers = {'head': 'o', 'tail': 's'}
        
        for part, (theta, x, y, r) in positions.items():
            if part == 'head':
                plt.scatter(x, y, color=colors.get(part, 'blue'), 
                           marker=markers.get(part, '^'), s=100, 
                           label=f'{part}: ({x:.2f}, {y:.2f})', zorder=5)
            elif part == 'tail':
                plt.scatter(x, y, color=colors.get(part, 'blue'), 
                           marker=markers.get(part, '^'), s=100, 
                           label=f'{part}: ({x:.2f}, {y:.2f})', zorder=5)
            else:
                plt.scatter(x, y, color='green', marker='^', s=60, alpha=0.7, zorder=4)
        
        # 连接龙的各部分
        x_coords = [pos[1] for pos in positions.values()]
        y_coords = [pos[2] for pos in positions.values()]
        plt.plot(x_coords, y_coords, 'g-', linewidth=2, alpha=0.7, label='龙身连接')
        
        plt.xlabel('X坐标 (m)')
        plt.ylabel('Y坐标 (m)')
        plt.title(f'问题3解答：最小螺距 = {min_pitch:.3f}m\n龙头到达调头空间边界时的位置分布')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def solve_problem3(self):
        """
        完整求解问题3
        """
        print("=== 问题3 完整解决方案 ===")
        print("目标：确定最小螺距，使得龙头前把手能够沿着螺线盘入到调头空间边界")
        
        # 寻找最小螺距
        min_pitch, results, positions = self.find_minimum_pitch()
        
        if min_pitch is not None:
            print(f"\n=== 最终答案 ===")
            print(f"最小螺距: {min_pitch:.3f} m")
            
            # 可视化结果
            self.visualize_solution(min_pitch, positions)
            
            # 保存结果
            results_df = pd.DataFrame(results)
            results_df.to_excel('results3.xlsx', index=False)
            print(f"详细结果已保存到 results3.xlsx")
            
            return min_pitch
        else:
            print("未找到可行的螺距解")
            return None

# 运行问题3求解
if __name__ == "__main__":
    solver = Problem3Solution()
    min_pitch = solver.solve_problem3()
    
    if min_pitch:
        print(f"\n=== 建模总结 ===")
        print(f"1. 核心约束：龙头前把手沿螺线运动到调头空间边界")
        print(f"2. 几何约束：龙的各部分之间固定距离关系")
        print(f"3. 边界条件：所有部分必须在半径4.5m的圆形区域内")
        print(f"4. 求解方法：数值搜索最小可行螺距")
        print(f"5. 最终结果：最小螺距 = {min_pitch:.3f} m")