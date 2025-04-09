import asyncio
import re
import logging
from typing import Optional, Tuple, Dict
import board
import busio
import time
import numpy as np
from adafruit_pca9685 import PCA9685
from pid import PID  # 自定义的PID类
import struct

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ServoController:
    """优化后的舵机控制器，支持软启动和频率校准"""
    def __init__(self):
        # 初始化I2C接口
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50  # SG90  50Hz
        self._current_angles = {0: 90, 3: 90}  # 记录当前角度
        self._locks: Dict[int, asyncio.Lock] = {
            0: asyncio.Lock(),
            3: asyncio.Lock()
        }  # 每个通道独立的异步锁
        
    def _angle_to_pulse(self, angle: float) -> int:
        """角度转PWM脉冲(0.5ms-2.5ms对应0-180度)"""
        pulse_us = 500 + (angle / 180.0) * 2000
        return int(pulse_us * 65535 / 20000)  # 20000us=50Hz周期
    
    async def set_angle(self, channel: int, target_angle: float, step_delay=0.03):
        """平滑移动到目标角度（添加异步锁防止竞态条件）"""
        async with self._locks[channel]:
            target_angle = max(15, min(165, target_angle))
            current = self._current_angles[channel]

            if abs(target_angle - current) < 1:
                return

            max_step = 5        # 单次最大移动步长
            angle_diff = target_angle - current
            if angle_diff > max_step:
                target_angle = current + max_step
            elif angle_diff < -max_step:
                target_angle = current - max_step
            steps = max(1, int(abs(target_angle - current) / 1 ))
            
            for angle in np.linspace(current, target_angle, steps):
                self.pca.channels[channel].duty_cycle = self._angle_to_pulse(angle)
                await asyncio.sleep(step_delay)
            self._current_angles[channel] = target_angle

    def deinit(self):
        """释放硬件资源"""
        self.pca.deinit()

class FaceTracker:
    def __init__(self):
        self.servo = ServoController()
        self.pid_vertical = PID(p=23.40, i=1.60, d=30.84, output_limits=(-20, 20))         # 垂直方向   
        self.pid_horizontal = PID(p=23.40, i=25.89, d=1.90, output_limits=(-20, 20))        # 水平方向
        self.queue = asyncio.Queue()
        self.x_buffer = []
        self.y_buffer = []
        self.buffer_size = 20        # 缓冲区大小
        self.dead_zone = 60          # 死区
        self.max_coord_change = 60      # 最大坐标变化
        # self.kalman_gain = 0.2      # 卡尔曼滤波增益
        
        # 添加振荡周期测量相关变量
        self.oscillation_detection = {
            'horizontal': {
                'prev_angle': 90,
                'prev_output': 0,
                'last_cross_time': 0,
                'cross_times': [],
                'periods': []
            },
            'vertical': {
                'prev_angle': 90,
                'prev_output': 0,
                'last_cross_time': 0,
                'cross_times': [],
                'periods': []
            }
        }
        
        # 记录开始执行时间
        self.start_time = time.time()

    async def initialize_servos(self):
        """初始化舵机到中心位置"""
        await self.servo.set_angle(0, 90)
        await self.servo.set_angle(3, 90)
        logging.info("舵机初始化完成")

    def detect_oscillation(self, axis, current_output, current_angle):
        """检测振荡并计算周期Tu"""
        data = self.oscillation_detection[axis]
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # 检测输出信号穿过零点（方向改变）
        if (data['prev_output'] < 0 and current_output > 0) or (data['prev_output'] > 0 and current_output < 0):
            # 记录穿越零点的时间
            if data['last_cross_time'] > 0:
                # 计算半周期
                half_period = current_time - data['last_cross_time']
                # 一个完整周期是两次穿越零点
                if len(data['cross_times']) >= 1:
                    full_period = current_time - data['cross_times'][-1]
                    data['periods'].append(full_period)
                    # 计算平均周期 (Tu)
                    avg_period = sum(data['periods']) / len(data['periods'])
                    logging.info(f"[{axis}] 检测到振荡! 当前周期(Tu): {full_period:.3f}秒, 平均周期: {avg_period:.3f}秒")
                
                data['cross_times'].append(current_time)
                # 只保留最近10个周期数据
                if len(data['cross_times']) > 10:
                    data['cross_times'].pop(0)
                if len(data['periods']) > 10:
                    data['periods'].pop(0)
            
            data['last_cross_time'] = current_time
            
        # 记录角度变化方向（用于检测振荡）
        if (data['prev_angle'] < 90 and current_angle > 90) or (data['prev_angle'] > 90 and current_angle < 90):
            if data['last_cross_time'] > 0:
                # 也可以用角度变化检测振荡
                logging.info(f"[{axis}] 舵机过中点: 当前角度={current_angle:.1f}, 时间={elapsed:.3f}秒")
        
        # 更新状态
        data['prev_output'] = current_output
        data['prev_angle'] = current_angle

    async def control_loop(self):
        CENTER_X = 640 / 2
        CENTER_Y = 640 / 2
        while True:
            try:
                # 处理所有队列数据
                while not self.queue.empty():
                    x, y = await self.queue.get()
                    # 更新缓冲区
                    self.x_buffer.append(x)
                    self.y_buffer.append(y)
                    if len(self.x_buffer) > self.buffer_size:
                        self.x_buffer.pop(0)
                        self.y_buffer.pop(0)

                if self.x_buffer:
                    x_smooth = sum(self.x_buffer) / len(self.x_buffer)
                    y_smooth = sum(self.y_buffer) / len(self.y_buffer)
                    # 归一化偏移量到[-1, 1]范围
                    norm_error_x = (x_smooth - CENTER_X) / CENTER_X
                    norm_error_y = (y_smooth - CENTER_Y) / CENTER_Y
                    
                    # 将归一化误差转换回实际误差范围
                    error_x = norm_error_x * CENTER_X
                    error_y = norm_error_y * CENTER_Y

                    if abs(error_x) >= self.dead_zone or abs(error_y) >= self.dead_zone:
                        vert_output = self.pid_vertical.compute(0, norm_error_y)  # 目标值为0
                        horz_output = self.pid_horizontal.compute(0, norm_error_x)  # 目标值为0
                        
                        # 记录详细的PID输出信息用于调试
                        logging.info(f"PID输出: vert={vert_output:.2f}, horz={horz_output:.2f}, "
                                    f"误差: norm_x={norm_error_x:.3f}, norm_y={norm_error_y:.3f}")
                        
                        # 计算目标角度
                        vert_angle = 90 - vert_output
                        horz_angle = 90 - horz_output
                        
                        # 检测振荡并计算周期
                        self.detect_oscillation('vertical', vert_output, vert_angle)
                        self.detect_oscillation('horizontal', horz_output, horz_angle)
                        
                        # 调整舵机方向符号（根据实际测试）
                        await asyncio.gather(
                            self.servo.set_angle(0, vert_angle),         # 垂直方向
                            self.servo.set_angle(3, horz_angle)          # 水平方向
                        )
            except Exception as e:
                logging.error(f"控制循环异常: {e}")
            await asyncio.sleep(0.005)

async def handle_connection(reader, writer, tracker: FaceTracker):
    """优化后的Socket数据处理(过滤空行)"""
    buffer = b""
    try:
        while True:
            data = await reader.read(4096)
            if not data:
                break
            buffer += data

            # 按换行符分割并过滤空行
            lines = buffer.split(b"\n")
            # 保留未处理的部分（最后一行可能不完整）
            buffer = lines[-1]
            # 处理所有完整行（排除空行）
            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    line_decoded = line.decode().strip()
                    x, y = map(float, line_decoded.split(','))
                    # 清空队列并放入最新数据
                    while not tracker.queue.empty():
                        tracker.queue.get_nowait()
                    tracker.queue.put_nowait((x, y))
                    # 添加日志记录
                    logging.info(f"接收坐标: {x}, {y}")  # <-- 关键修复点
                except ValueError:
                    logging.warning(f"无效数据格式: {line}")
                except Exception as e:
                    logging.error(f"解析异常: {e}")
    except ConnectionResetError:
        logging.warning("客户端连接已关闭")
    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    # 修改日志级别为INFO，确保能看到所有调试信息
    logging.getLogger().setLevel(logging.INFO)
    
    # 创建PID参数自动调整测试模式
    tune_mode = input("是否启用PID调参模式? (y/n): ").strip().lower() == 'y'
    
    tracker = FaceTracker()
    
    # 如果启用调参模式，按照Ziegler-Nichols方法逐步增加Kp
    if tune_mode:
        kp_start = float(input("输入起始Kp值 (建议5.0): ") or "5.0")
        kp_step = float(input("输入Kp增量步长 (建议2.0): ") or "2.0")
        
        tracker.pid_vertical = PID(p=kp_start, i=0.0, d=0.0, output_limits=(-20, 20))
        tracker.pid_horizontal = PID(p=kp_start, i=0.0, d=0.0, output_limits=(-20, 20))
        
        logging.info(f"PID调参模式已启动! 初始Kp={kp_start}, 步长={kp_step}")
        logging.info("请观察日志中的振荡周期Tu数据,当系统开始持续振荡时记录下Ku和Tu值")
        
        # 创建自动调参任务
        async def auto_tune():
            while True:
                await asyncio.sleep(10)  # 每10秒增加一次Kp
                current_kp = tracker.pid_vertical.p
                new_kp = current_kp + kp_step
                tracker.pid_vertical.p = new_kp
                tracker.pid_horizontal.p = new_kp
                logging.info(f"自动增加Kp值: {current_kp} -> {new_kp}")
        
        tune_task = asyncio.create_task(auto_tune())
    
    await tracker.initialize_servos()
    server = await asyncio.start_server(
        lambda r, w: handle_connection(r, w, tracker),
        '0.0.0.0', 8888
    )
    logging.info("服务器已启动，等待连接...")
    
    try:
        async with server:
            if tune_mode:
                await asyncio.gather(
                    server.serve_forever(),
                    tracker.control_loop(),
                    tune_task
                )
            else:
                await asyncio.gather(
                    server.serve_forever(),
                    tracker.control_loop()
                )
    finally:
        if tune_mode and 'tune_task' in locals():
            tune_task.cancel()
        tracker.servo.deinit()  # 确保资源释放
        
        # 如果有收集到周期数据，显示计算建议
        if tune_mode and (tracker.oscillation_detection['horizontal']['periods'] or 
                         tracker.oscillation_detection['vertical']['periods']):
            # 计算平均周期
            h_periods = tracker.oscillation_detection['horizontal']['periods']
            v_periods = tracker.oscillation_detection['vertical']['periods']
            
            if h_periods:
                h_tu = sum(h_periods) / len(h_periods)
                logging.info(f"水平方向平均振荡周期Tu = {h_tu:.3f}秒")
                logging.info(f"建议参数 (水平方向):")
                logging.info(f"  P控制: Kp = {0.5 * tracker.pid_horizontal.p:.2f}")
                logging.info(f"  PI控制: Kp = {0.45 * tracker.pid_horizontal.p:.2f}, Ki = {0.54 * 0.45 * tracker.pid_horizontal.p / h_tu:.2f}")
                logging.info(f"  PID控制: Kp = {0.6 * tracker.pid_horizontal.p:.2f}, "
                            f"Ki = {1.2 * 0.6 * tracker.pid_horizontal.p / h_tu:.2f}, "
                            f"Kd = {0.075 * 0.6 * tracker.pid_horizontal.p * h_tu:.2f}")
            
            if v_periods:
                v_tu = sum(v_periods) / len(v_periods)
                logging.info(f"垂直方向平均振荡周期Tu = {v_tu:.3f}秒")
                logging.info(f"建议参数 (垂直方向):")
                logging.info(f"  P控制: Kp = {0.5 * tracker.pid_vertical.p:.2f}")
                logging.info(f"  PI控制: Kp = {0.45 * tracker.pid_vertical.p:.2f}, Ki = {0.54 * 0.45 * tracker.pid_vertical.p / v_tu:.2f}")
                logging.info(f"  PID控制: Kp = {0.6 * tracker.pid_vertical.p:.2f}, "
                            f"Ki = {1.2 * 0.6 * tracker.pid_vertical.p / v_tu:.2f}, "
                            f"Kd = {0.075 * 0.6 * tracker.pid_vertical.p * v_tu:.2f}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server shutdown")
