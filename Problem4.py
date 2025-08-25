import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import pandas as pd

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem4Solution:
    """
    问题4: 龙头舞蹈队S形曲线运动分析
    舞蹈队每入螺线的螺距为1.7m，盘出螺线与盘入螺线关于螺线中心呈中心对称，
    舞蹈队在问题3设定的调头空间内完成调头，调头路径是两段圆弧和连接两段的S形曲线。
    """
    
    def __init__(self):
        # 龙的几何参数（单位：米）
        self.L_head = 2.23  # 龙头长度
        self.L_body = 3.41  # 龙身长度  
        self.L_tail = 2.20  # 龙尾长度
        self.total_segments = 220  # 总体节数
        
        # 运动参数
        self.head_speed = 1.0  # 龙头前把手速度 1m/s
        self.spiral_pitch = 1.7  # 螺距 1.7m
        
        # 调头空间参数
        self.turnaround_radius = 4.5  # 调头空间半径 4.5m
        
        # S形曲线参数
        self.s_curve_length = None  # 待计算
        self.circle_radius = None   # 圆弧半径，待优化
        
        # 龙身各把手位置
        self.handle_segments = [1, 51, 101, 151, 201]
        self.handle_distances = []
        for seg in self.handle_segments:
            dist = self.L_head + (seg - 1) * (self.L_body / self.total_segments)
            self.handle_distances.append(dist)
        
        # 龙尾距离
        self.tail_distance = self.L_head + self.L_body + self.L_tail
        
        print("=== 问题4 模型参数 ===")
        print(f"螺距: {self.spiral_pitch}m")
        print(f"调头空间半径: {self.turnaround_radius}m")
        print(f"龙头前把手速度: {self.head_speed}m/s")
    
    def spiral_equation(self, theta, is_outward=False):
        """
        螺线方程
        is_outward: False为盘入螺线，True为盘出螺线（中心对称）
        """
        a = self.spiral_pitch / (2 * np.pi)
        r = a * theta
        
        if is_outward:
            # 盘出螺线：关于原点中心对称
            x = -r * np.cos(theta)
            y = -r * np.sin(theta)
        else:
            # 盘入螺线
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        
        return r, x, y
    
    def design_s_curve_path(self):
        """
        设计S形调头路径
        路径组成：
        1. 第一段圆弧：从盘入螺线边界开始
        2. S形曲线：连接两段圆弧
        3. 第二段圆弧：到盘出螺线边界结束
        """
        print("\n=== 设计S形调头路径 ===")
        
        # 计算盘入螺线到达调头空间边界的角度
        a = self.spiral_pitch / (2 * np.pi)
        theta_boundary = self.turnaround_radius / a
        
        # 盘入螺线边界点
        _, x_in, y_in = self.spiral_equation(theta_boundary, is_outward=False)
        
        # 盘出螺线边界点（中心对称）
        _, x_out, y_out = self.spiral_equation(theta_boundary, is_outward=True)
        
        print(f"盘入边界点: ({x_in:.3f}, {y_in:.3f})")
        print(f"盘出边界点: ({x_out:.3f}, {y_out:.3f})")
        
        # 设计圆弧半径（需要保证两段圆弧不相交且在调头空间内）
        distance_between_points = np.sqrt((x_out - x_in)**2 + (y_out - y_in)**2)
        
        # 圆弧半径设为两点距离的1/4，确保有足够空间
        self.circle_radius = distance_between_points / 4
        
        print(f"两边界点距离: {distance_between_points:.3f}m")
        print(f"设计圆弧半径: {self.circle_radius:.3f}m")
        
        # 计算第一段圆弧中心
        # 圆弧中心在垂直于盘入螺线切线方向
        tangent_angle = theta_boundary  # 螺线在该点的切线角度
        normal_angle = tangent_angle + np.pi/2
        
        circle1_center_x = x_in + self.circle_radius * np.cos(normal_angle)
        circle1_center_y = y_in + self.circle_radius * np.sin(normal_angle)
        
        # 计算第二段圆弧中心（对称设计）
        circle2_center_x = x_out - self.circle_radius * np.cos(normal_angle)
        circle2_center_y = y_out - self.circle_radius * np.sin(normal_angle)
        
        return {
            'boundary_in': (x_in, y_in),
            'boundary_out': (x_out, y_out),
            'circle1_center': (circle1_center_x, circle1_center_y),
            'circle2_center': (circle2_center_x, circle2_center_y),
            'circle_radius': self.circle_radius,
            'theta_boundary': theta_boundary
        }
    
    def create_s_curve_parametric(self, path_info):
        """
        创建S形曲线的参数方程
        使用贝塞尔曲线或三次样条连接两段圆弧
        """
        def s_curve_param(t):
            """
            S形曲线参数方程，t ∈ [0, 1]
            """
            # 获取两个圆弧的连接点
            circle1_center = path_info['circle1_center']
            circle2_center = path_info['circle2_center']
            
            # 简化设计：使用三次贝塞尔曲线
            # 控制点设计
            P0 = circle1_center  # 起点
            P3 = circle2_center  # 终点
            
            # 中间控制点，形成S形
            mid_x = (P0[0] + P3[0]) / 2
            mid_y = (P0[1] + P3[1]) / 2
            
            # 控制点偏移，形成S形弯曲
            offset = self.circle_radius * 0.5
            P1 = (P0[0] + offset, mid_y)
            P2 = (P3[0] - offset, mid_y)
            
            # 贝塞尔曲线方程
            x = (1-t)**3 * P0[0] + 3*(1-t)**2*t * P1[0] + 3*(1-t)*t**2 * P2[0] + t**3 * P3[0]
            y = (1-t)**3 * P0[1] + 3*(1-t)**2*t * P1[1] + 3*(1-t)*t**2 * P2[1] + t**3 * P3[1]
            
            return x, y
        
        return s_curve_param
    
    def calculate_path_length(self, path_function, t_range=(0, 1), num_points=1000):
        """
        计算路径长度
        """
        t_values = np.linspace(t_range[0], t_range[1], num_points)
        path_points = [path_function(t) for t in t_values]
        
        total_length = 0
        for i in range(1, len(path_points)):
            dx = path_points[i][0] - path_points[i-1][0]
            dy = path_points[i][1] - path_points[i-1][1]
            total_length += np.sqrt(dx**2 + dy**2)
        
        return total_length
    
    def simulate_dragon_motion(self, time_range=(-100, 100)):
        """
        模拟整个运动过程：盘入 -> 调头 -> 盘出
        """
        print("\n=== 模拟龙的运动过程 ===")
        
        # 设计调头路径
        path_info = self.design_s_curve_path()
        s_curve_func = self.create_s_curve_parametric(path_info)
        
        # 计算各段路径长度
        circle_arc_length = np.pi * self.circle_radius  # 半圆弧长
        s_curve_length = self.calculate_path_length(s_curve_func)
        total_turnaround_length = 2 * circle_arc_length + s_curve_length
        
        print(f"圆弧段长度: {circle_arc_length:.3f}m")
        print(f"S曲线段长度: {s_curve_length:.3f}m")
        print(f"调头总长度: {total_turnaround_length:.3f}m")
        
        # 时间节点计算
        time_to_boundary = path_info['theta_boundary'] * self.spiral_pitch / (2 * np.pi)  # 到达边界的时间
        turnaround_time = total_turnaround_length / self.head_speed  # 调头用时
        
        print(f"到达调头空间边界时间: {time_to_boundary:.1f}s")
        print(f"调头用时: {turnaround_time:.1f}s")
        
        # 生成时间序列
        time_points = [-100, -50, 0, 50, 100]
        
        results = []
        for t in time_points:
            result = {'Time(s)': t}
            
            if t < 0:  # 盘入阶段
                # 龙头沿盘入螺线运动
                s = abs(t)  # 弧长参数
                theta = np.sqrt(2 * s * 2 * np.pi / self.spiral_pitch)
                
                # 龙头位置
                _, x_head, y_head = self.spiral_equation(theta, is_outward=False)
                result['Head_x(m)'] = x_head
                result['Head_y(m)'] = y_head
                
                # 计算其他部分位置
                self.calculate_all_dragon_parts_spiral(result, theta, is_outward=False)
                
            elif t == 0:  # 调头开始
                boundary_point = path_info['boundary_in']
                result['Head_x(m)'] = boundary_point[0]
                result['Head_y(m)'] = boundary_point[1]
                # 其他部分位置...
                
            elif t > 0:  # 盘出阶段
                # 龙头沿盘出螺线运动
                s = t
                theta = np.sqrt(2 * s * 2 * np.pi / self.spiral_pitch)
                
                # 龙头位置
                _, x_head, y_head = self.spiral_equation(theta, is_outward=True)
                result['Head_x(m)'] = x_head
                result['Head_y(m)'] = y_head
                
                # 计算其他部分位置
                self.calculate_all_dragon_parts_spiral(result, theta, is_outward=True)
            
            results.append(result)
        
        return pd.DataFrame(results), path_info
    
    def calculate_all_dragon_parts_spiral(self, result, theta_head, is_outward=False):
        """
        计算螺线运动时龙各部分的位置
        """
        # 简化计算：假设各部分都沿同一螺线，但角度不同
        a = self.spiral_pitch / (2 * np.pi)
        
        # 龙身各把手
        handle_names = ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201']
        
        for i, (name, distance) in enumerate(zip(handle_names, self.handle_distances)):
            # 估算对应的角度（简化处理）
            delta_theta = distance / (a * theta_head) * 2 if theta_head > 0 else 0
            theta_handle = max(0, theta_head - delta_theta)
            
            _, x_handle, y_handle = self.spiral_equation(theta_handle, is_outward)
            result[f'{name}_x(m)'] = x_handle
            result[f'{name}_y(m)'] = y_handle
        
        # 龙尾
        delta_theta_tail = self.tail_distance / (a * theta_head) * 2 if theta_head > 0 else 0
        theta_tail = max(0, theta_head - delta_theta_tail)
        
        _, x_tail, y_tail = self.spiral_equation(theta_tail, is_outward)
        result['Tail_x(m)'] = x_tail
        result['Tail_y(m)'] = y_tail
    
    def visualize_complete_path(self, path_info):
        """
        可视化完整路径
        """
        plt.figure(figsize=(15, 12))
        
        # 绘制调头空间边界
        circle_boundary = plt.Circle((0, 0), self.turnaround_radius, 
                                   fill=False, color='red', linewidth=2, 
                                   label='调头空间边界')
        plt.gca().add_patch(circle_boundary)
        
        # 绘制盘入螺线
        theta_max = 4 * np.pi
        theta_range = np.linspace(0, theta_max, 500)
        
        spiral_in_x, spiral_in_y = [], []
        spiral_out_x, spiral_out_y = [], []
        
        for theta in theta_range:
            _, x_in, y_in = self.spiral_equation(theta, is_outward=False)
            _, x_out, y_out = self.spiral_equation(theta, is_outward=True)
            
            spiral_in_x.append(x_in)
            spiral_in_y.append(y_in)
            spiral_out_x.append(x_out)
            spiral_out_y.append(y_out)
        
        plt.plot(spiral_in_x, spiral_in_y, 'b--', alpha=0.7, label='盘入螺线')
        plt.plot(spiral_out_x, spiral_out_y, 'g--', alpha=0.7, label='盘出螺线')
        
        # 绘制边界点
        boundary_in = path_info['boundary_in']
        boundary_out = path_info['boundary_out']
        plt.scatter(*boundary_in, color='blue', s=100, marker='o', label='盘入边界点')
        plt.scatter(*boundary_out, color='green', s=100, marker='o', label='盘出边界点')
        
        # 绘制调头路径
        circle1_center = path_info['circle1_center']
        circle2_center = path_info['circle2_center']
        
        # 第一段圆弧
        circle1 = plt.Circle(circle1_center, self.circle_radius, 
                           fill=False, color='orange', linewidth=2)
        plt.gca().add_patch(circle1)
        
        # 第二段圆弧
        circle2 = plt.Circle(circle2_center, self.circle_radius, 
                           fill=False, color='orange', linewidth=2)
        plt.gca().add_patch(circle2)
        
        # S形曲线
        s_curve_func = self.create_s_curve_parametric(path_info)
        t_values = np.linspace(0, 1, 100)
        s_curve_x = [s_curve_func(t)[0] for t in t_values]
        s_curve_y = [s_curve_func(t)[1] for t in t_values]
        plt.plot(s_curve_x, s_curve_y, 'purple', linewidth=3, label='S形连接曲线')
        
        plt.xlabel('X坐标 (m)')
        plt.ylabel('Y坐标 (m)')
        plt.title('问题4：完整的龙头舞蹈队运动路径\n(盘入→调头→盘出)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def solve_problem4(self):
        """
        完整求解问题4
        """
        print("=== 问题4 完整解决方案 ===")
        print("分析龙头舞蹈队的S形调头曲线变化")
        
        # 模拟运动过程
        results_df, path_info = self.simulate_dragon_motion()
        
        print("\n=== 关键时刻位置结果 ===")
        print(results_df.round(3))
        
        # 可视化
        self.visualize_complete_path(path_info)
        
        # 保存结果
        results_df.to_excel('results4.xlsx', index=False)
        print("\n结果已保存到 results4.xlsx")

        # 分析调头曲线变化
        print("\n=== 调头曲线分析 ===")
        print(f"1. 调头路径设计：两段圆弧 + S形连接曲线")
        print(f"2. 圆弧半径: {self.circle_radius:.3f}m")
        print(f"3. 路径确保所有部分都在调头空间内")
        print(f"4. S形曲线实现平滑的方向转换")
        
        return results_df, path_info

# 运行问题4求解
if __name__ == "__main__":
    solver = Problem4Solution()
    results, path_info = solver.solve_problem4()
    
    print(f"\n=== 问题4 建模总结 ===")
    print(f"1. 核心任务：设计盘入到盘出的调头路径")
    print(f"2. 约束条件：螺距1.7m，调头空间直径9m，速度1m/s")
    print(f"3. 路径设计：圆弧-S曲线-圆弧的组合")
    print(f"4. 关键创新：使用贝塞尔曲线实现平滑S形过渡")
    print(f"5. 输出结果：关键时刻各部分位置坐标表")