import time
class PID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 1.00   #間隔時間
        self.current_time = time.time()
        self.last_time = self.current_time

        self.SetPoint = 0.0 ##須達成的目標
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0

        self.I_max_modify = 20.0
        self.output = 0.0
    def update(self, feedback_value):
        # 誤差
        error = self.SetPoint - feedback_value
        self.current_time = time.time()
        # 間隔時間
        delta_time = self.current_time - self.last_time
        if(delta_time >= self.sample_time):
            self.ITerm += error * delta_time
            # 限制積分項   最大最小值(可要可不要???)
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
 