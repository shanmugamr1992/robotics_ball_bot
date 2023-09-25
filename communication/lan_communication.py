from . import lan

class LanCommunication:
    def __init__(self, ip, port):
        lan.stop()
        lan.start(ip, port, frame_shape=(360, 640))  

    def get_sensor_readings(self):
        return lan.read()[1:]

    def get_camera_pic(self):
        return lan.get_frame() 
    
    def send_motor_vals(self, motor_vals):
        lan.write(motor_vals)