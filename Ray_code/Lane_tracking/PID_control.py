import time
class PID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.1  #間隔時間
        self.current_time = time.time()
        self.last_time = self.current_time

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0

    def update(self, feedback_value):
        # 誤差
        error = feedback_value
        self.current_time = time.time()
        # 間隔時間
        delta_time = self.current_time - self.last_time
        if(delta_time >= self.sample_time):
            self.ITerm += error * delta_time
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = (error - self.last_error) / delta_time
            # 更新時間
            self.last_time = self.current_time
            self.last_error = error
            # PID结果
            return (self.Kp * error) + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
 