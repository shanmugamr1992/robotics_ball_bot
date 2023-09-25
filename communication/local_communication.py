from .device import RealsenseCamera
from .vex_serial import VexCortex

class LocalCommunication:
    def __init__(self):
        self.cam = RealsenseCamera()
        self.bot = VexCortex("/dev/ttyUSB0")

    def get_sensor_readings(self):
        return self.bot.sensors()

    def get_camera_pic(self):
        return self.cam.read()
    
    def send_motor_vals(self, motor_vals):
        self.bot._motor_values.set(motor_vals)
