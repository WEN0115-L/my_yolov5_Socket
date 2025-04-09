class PID:
    def __init__(self, p=1.0, i=0.0, d=0.0, output_limits=(None, None)):
        self.p = p
        self.i = i
        self.d = d
        self._integral = 0.0
        self._prev_error = 0.0
        self.output_limits = output_limits
        
    def compute(self, setpoint: float, measurement: float) -> float:
        error = setpoint - measurement
        output = self.p * error + self.i * self._integral + self.d * (error - self._prev_error)
        
        # 暂存未限幅的输出用于积分判断
        temp_output = output
        
        # 输出限幅
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)
        
        # 抗积分饱和：仅当输出未饱和时更新积分
        if temp_output == output:
            self._integral += error * 0.02  # 控制周期50Hz对应0.02s
        self._prev_error = error
        
        return output