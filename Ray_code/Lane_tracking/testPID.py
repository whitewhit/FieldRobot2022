import time
class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):
        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time
        self.clear()
    # 清理係數
    def clear(self):
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0

        self.I_max_modify = 20.0
        self.output = 0.0
    def update(self, feedback_value, current_time=None):
        # 誤差
        error = self.SetPoint - feedback_value
        self.current_time = current_time if current_time is not None else time.time()
        # 間隔時間
        delta_time = self.current_time - self.last_time
        if(delta_time >= self.sample_time):
            self.ITerm += error * delta_time
            # 限制積分項   最大最小值
            if(self.ITerm < -self.I_max_modify):
                self.ITerm = -self.I_max_modify
            elif(self.ITerm > self.I_max_modify):
                self.ITerm = self.I_max_modify
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = (error - self.last_error) / delta_time
            # 更新時間
            self.last_time = self.current_time
            self.last_error = error
            # PID结果
            self.output = (self.Kp * error) + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
 