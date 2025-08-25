import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
from scipy.integrate import quad
import pandas as pd
from scipy.interpolate import interp1d

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem5Solution:
    """
    问题5: 龙头最大行进速度优化分析
    在问题4设定的路径行进过程中，龙头行进速度保持不变，
    请确定龙头的最大行进速度，使得舞蹈队各把手的速度均不超过2 m/s。
    """
    
    def __init__(self):
        # 龙的几何参数（单位：米）
        self.L_head = 2.23  # 龙头长度
        self.L_body = 3.41  # 龙身长度  
        self.L_tail = 2.20  # 龙尾长度
        self.total_segments = 220  # 总体节数
        
        # 路径参数
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
        
        print("=== 问题5 模型参数 ===")
        print(f"螺距: {self.spiral_pitch}m")
        print(f"调头空间半径: {self.turnaround_radius}m")
        print(f"各把手最大允许速度: {self.max_handle_speed}m/s")
        print(f"把手距离: {[f'{d:.3f}m' for d in self.handle_distances]}")
        print(f"龙尾距离: {self.tail_distance:.3f}m")
    
    def spiral_equation(self, theta, is_outward=False):
        """
        螺线方程
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
    
    def spiral_curvature(self, theta):
        """
        计算螺线在给定角度处的曲率
        对于阿基米德螺线 r = a*θ，曲率公式为：
        κ = (r² + 2(dr/dθ)² - r(d²r/dθ²)) / (r² + (dr/dθ)²)^(3/2)
        """
        a = self.spiral_pitch / (2 * np.pi)
        r = a * theta
        
        if theta == 0:
            return float('inf')  # 在原点处曲率无限大
        
        dr_dtheta = a
        d2r_dtheta2 = 0  # 对于阿基米德螺线，二阶导数为0
        
        numerator = r**2 + 2 * dr_dtheta**2 - r * d2r_dtheta2
        denominator = (r**2 + dr_dtheta**2)**(3/2)
        
        curvature = numerator / denominator if denominator > 0 else 0
        return curvature
    
    def calculate_speed_amplification_factor(self, theta, distance_behind):
        """
        计算速度放大因子
        当龙头沿螺线运动时，后面的把手由于曲率效应会有速度放大
        """
        if theta <= 0:
            return 1.0
        
        # 计算曲率
        curvature = self.spiral_curvature(theta)
        
        # 速度放大因子的近似公式
        # 基于曲线运动学，内侧点的速度会相对放大
        a = self.spiral_pitch / (2 * np.pi)
        r = a * theta
        
        # 考虑螺线的特殊性质
        if r > 0:
            # 基于几何关系的速度放大因子
            amplification = 1 + distance_behind * curvature / (2 * r)
            # 限制放大因子在合理范围内
            amplification = max(1.0, min(amplification, 5.0))
        else:
            amplification = 1.0
        
        return amplification
    
    def calculate_handle_speeds_spiral(self, head_speed, theta):
        """
        计算螺线运动时各把手的速度
        """
        speeds = {}
        speeds['Head'] = head_speed
        
        # 计算各把手速度
        handle_names = ['Body_Seg1', 'Body_Seg51', 'Body_Seg101', 'Body_Seg151', 'Body_Seg201']
        
        for i, (name, distance) in enumerate(zip(handle_names, self.handle_distances)):
            # 计算该把手对应的角度位置
            a = self.spiral_pitch / (2 * np.pi)
            
            # 简化计算：假设把手距离对应的角度差
            if theta > 0:
                # 基于弧长关系估算角度差
                arc_length_head = a * theta**2 / 2  # 螺线弧长近似
                theta_handle = max(0, theta - distance / a)  # 简化的角度估算
            else:
                theta_handle = 0
            
            # 计算速度放大因子
            amplification = self.calculate_speed_amplification_factor(theta_handle, distance)
            
            # 把手速度
            handle_speed = head_speed * amplification
            speeds[name] = handle_speed
        
        # 龙尾速度
        tail_amplification = self.calculate_speed_amplification_factor(
            max(0, theta - self.tail_distance / (self.spiral_pitch / (2 * np.pi))), 
            self.tail_distance
        )
        speeds['Tail'] = head_speed * tail_amplification
        
        return speeds
    
    def find_maximum_speed_at_position(self, theta):
        """
        在给定螺线位置找到满足约束的最大龙头速度
        """
        def speed_constraint_violation(head_speed):
            speeds = self.calculate_handle_speeds_spiral(head_speed, theta)
            max_handle_speed = max([v for k, v in speeds.items() if k != 'Head'])
            return max_handle_speed - self.max_handle_speed
        
        # 二分搜索找到最大可行速度
        low_speed = 0.1
        high_speed = 5.0
        
        # 检查高速度是否可行
        if speed_constraint_violation(high_speed) <= 0:
            return high_speed
        
        # 二分搜索
        for _ in range(50):  # 最多迭代50次
            mid_speed = (low_speed + high_speed) / 2
            
            if speed_constraint_violation(mid_speed) <= 0:
                low_speed = mid_speed
            else:
                high_speed = mid_speed
            
            if abs(high_speed - low_speed) < 0.001:
                break
        
        return low_speed
    
    def analyze_speed_along_spiral(self):
        """
        分析沿螺线不同位置的最大允许速度
        """
        print("\n=== 分析沿螺线的最大允许速度 ===")
        
        # 计算螺线从原点到调头空间边界的角度范围
        a = self.spiral_pitch / (2 * np.pi)
        theta_boundary = self.turnaround_radius / a
        
        print(f"边界角度: {theta_boundary:.3f} 弧度 ({theta_boundary*180/np.pi:.1f}度)")
        
        # 在螺线上取多个点进行分析
        theta_points = np.linspace(0.5, theta_boundary, 20)  # 避免theta=0的奇点
        
        results = []
        for theta in theta_points:
            max_speed = self.find_maximum_speed_at_position(theta)
            r, x, y = self.spiral_equation(theta)
            
            # 计算各把手在该位置的速度
            speeds = self.calculate_handle_speeds_spiral(max_speed, theta)
            
            result = {
                'theta': theta,
                'radius': r,
                'x': x,
                'y': y,
                'max_head_speed': max_speed,
                'curvature': self.spiral_curvature(theta)
            }
            
            # 添加各把手速度
            for name, speed in speeds.items():
                result[f'{name}_speed'] = speed
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def find_global_maximum_speed(self):
        """
        找到整个路径上的全局最大允许速度
        """
        print("\n=== 寻找全局最大允许速度 ===")
        
        # 分析螺线段
        spiral_analysis = self.analyze_speed_along_spiral()
        
        # 找到螺线段的最小最大速度（这就是瓶颈）
        min_max_speed_spiral = spiral_analysis['max_head_speed'].min()
        bottleneck_idx = spiral_analysis['max_head_speed'].idxmin()
        bottleneck_info = spiral_analysis.iloc[bottleneck_idx]
        
        print(f"螺线段速度瓶颈:")
        print(f"  位置: θ={bottleneck_info['theta']:.3f}, r={bottleneck_info['radius']:.3f}m")
        print(f"  坐标: ({bottleneck_info['x']:.3f}, {bottleneck_info['y']:.3f})")
        print(f"  最大允许龙头速度: {min_max_speed_spiral:.3f} m/s")
        print(f"  曲率: {bottleneck_info['curvature']:.6f} 1/m")
        
        # 调头段分析（简化处理）
        # 调头段包括圆弧和S曲线，通常曲率更大，速度约束更严格
        turnaround_max_speed = self.analyze_turnaround_speed()
        
        print(f"\n调头段最大允许速度: {turnaround_max_speed:.3f} m/s")
        
        # 全局最大速度是所有段的最小值
        global_max_speed = min(min_max_speed_spiral, turnaround_max_speed)
        
        print(f"\n=== 最终结果 ===")
        print(f"全局最大允许龙头速度: {global_max_speed:.3f} m/s")
        
        return global_max_speed, spiral_analysis, bottleneck_info
    
    def analyze_turnaround_speed(self):
        """
        分析调头段的最大允许速度
        调头段包括圆弧和S形曲线，通常是速度的主要约束
        """
        # 估算调头段的典型曲率半径
        # 基于问题4的设计，圆弧半径约为两边界点距离的1/4
        
        # 计算边界点
        a = self.spiral_pitch / (2 * np.pi)
        theta_boundary = self.turnaround_radius / a
        _, x_in, y_in = self.spiral_equation(theta_boundary, is_outward=False)
        _, x_out, y_out = self.spiral_equation(theta_boundary, is_outward=True)
        
        distance_between_points = np.sqrt((x_out - x_in)**2 + (y_out - y_in)**2)
        circle_radius = distance_between_points / 4
        
        # 圆弧段的曲率
        circle_curvature = 1 / circle_radius
        
        # S形曲线的最大曲率（估算）
        s_curve_max_curvature = 2 / circle_radius  # 通常是圆弧曲率的2倍
        
        # 基于曲率计算最大允许速度
        # 使用最严格的约束（最大曲率点）
        max_curvature = max(circle_curvature, s_curve_max_curvature)
        
        # 对于调头段，龙尾的速度放大最明显
        max_amplification = 1 + self.tail_distance * max_curvature / 2
        
        # 计算最大允许龙头速度
        turnaround_max_speed = self.max_handle_speed / max_amplification
        
        print(f"调头段分析:")
        print(f"  圆弧半径: {circle_radius:.3f}m")
        print(f"  最大曲率: {max_curvature:.6f} 1/m")
        print(f"  最大速度放大因子: {max_amplification:.3f}")
        
        return turnaround_max_speed
    
    def visualize_speed_analysis(self, spiral_analysis, global_max_speed):
        """
        可视化速度分析结果
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 螺线路径上的最大允许速度分布
        ax1.plot(spiral_analysis['theta'], spiral_analysis['max_head_speed'], 'b-o', linewidth=2)
        ax1.axhline(y=global_max_speed, color='r', linestyle='--', 
                   label=f'全局最大速度 = {global_max_speed:.3f} m/s')
        ax1.set_xlabel('角度 θ (弧度)')
        ax1.set_ylabel('最大允许龙头速度 (m/s)')
        ax1.set_title('沿螺线的最大允许速度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 曲率分布
        ax2.plot(spiral_analysis['theta'], spiral_analysis['curvature'], 'g-o', linewidth=2)
        ax2.set_xlabel('角度 θ (弧度)')
        ax2.set_ylabel('曲率 (1/m)')
        ax2.set_title('螺线曲率分布')
        ax2.grid(True, alpha=0.3)
        
        # 3. 螺线轨迹和速度标注
        ax3.plot(spiral_analysis['x'], spiral_analysis['y'], 'b-', linewidth=2, label='螺线轨迹')
        
        # 标注速度瓶颈点
        bottleneck_idx = spiral_analysis['max_head_speed'].idxmin()
        bottleneck = spiral_analysis.iloc[bottleneck_idx]
        ax3.scatter(bottleneck['x'], bottleneck['y'], color='red', s=200, marker='*', 
                   label=f'速度瓶颈点: {bottleneck["max_head_speed"]:.3f} m/s')
        
        # 调头空间边界
        circle = plt.Circle((0, 0), self.turnaround_radius, fill=False, color='red', 
                           linestyle='--', label='调头空间边界')
        ax3.add_patch(circle)
        
        ax3.set_xlabel('X坐标 (m)')
        ax3.set_ylabel('Y坐标 (m)')
        ax3.set_title('螺线轨迹与速度分析')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # 4. 各把手速度对比（在瓶颈点）
        bottleneck_speeds = self.calculate_handle_speeds_spiral(
            bottleneck['max_head_speed'], bottleneck['theta']
        )
        
        handles = list(bottleneck_speeds.keys())
        speeds = list(bottleneck_speeds.values())
        
        colors = ['red' if handle == 'Head' else 'blue' for handle in handles]
        bars = ax4.bar(range(len(handles)), speeds, color=colors, alpha=0.7)
        
        ax4.axhline(y=self.max_handle_speed, color='red', linestyle='--', 
                   label=f'速度上限 = {self.max_handle_speed} m/s')
        ax4.set_xlabel('把手位置')
        ax4.set_ylabel('速度 (m/s)')
        ax4.set_title(f'瓶颈点各把手速度分布\n(龙头速度 = {bottleneck["max_head_speed"]:.3f} m/s)')
        ax4.set_xticks(range(len(handles)))
        ax4.set_xticklabels(handles, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def solve_problem5(self):
        """
        完整求解问题5
        """
        print("=== 问题5 完整解决方案 ===")
        print("确定龙头的最大行进速度，使得舞蹈队各把手的速度均不超过2 m/s")
        
        # 寻找全局最大速度
        global_max_speed, spiral_analysis, bottleneck_info = self.find_global_maximum_speed()
        
        # 验证结果
        print(f"\n=== 结果验证 ===")
        speeds_at_max = self.calculate_handle_speeds_spiral(
            global_max_speed, bottleneck_info['theta']
        )
        
        print("在最大速度下各把手的速度:")
        for handle, speed in speeds_at_max.items():
            status = "✓" if speed <= self.max_handle_speed else "✗"
            print(f"  {handle}: {speed:.3f} m/s {status}")
        
        # 可视化分析
        self.visualize_speed_analysis(spiral_analysis, global_max_speed)
        
        # 保存详细结果
        spiral_analysis.to_excel('results5.xlsx', index=False)
        
        # 创建总结报告
        summary = {
            'Global_Max_Head_Speed(m/s)': [global_max_speed],
            'Bottleneck_Position_theta': [bottleneck_info['theta']],
            'Bottleneck_Position_x(m)': [bottleneck_info['x']],
            'Bottleneck_Position_y(m)': [bottleneck_info['y']],
            'Bottleneck_Curvature(1/m)': [bottleneck_info['curvature']],
            'Max_Handle_Speed_Limit(m/s)': [self.max_handle_speed]
        }
        
        for handle, speed in speeds_at_max.items():
            summary[f'{handle}_Speed_at_Max(m/s)'] = [speed]
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_excel('results5_summary.xlsx', index=False)

        print(f"\n详细分析结果已保存到:")
        print(f"  - results5.xlsx (螺线分析)")
        print(f"  - results5_summary.xlsx (总结报告)")

        return global_max_speed, spiral_analysis

# 运行问题5求解
if __name__ == "__main__":
    solver = Problem5Solution()
    max_speed, analysis = solver.solve_problem5()
    
    print(f"\n=== 问题5 最终答案 ===")
    print(f"龙头的最大行进速度: {max_speed:.3f} m/s")
    
    print(f"\n=== 建模总结 ===")
    print(f"1. 约束条件：所有把手速度不超过2 m/s")
    print(f"2. 关键因素：螺线曲率导致的速度放大效应")
    print(f"3. 瓶颈位置：螺线中曲率较大的区域")
    print(f"4. 求解方法：基于曲率分析的速度优化")
    print(f"5. 实际意义：为龙舞表演提供安全速度指导")