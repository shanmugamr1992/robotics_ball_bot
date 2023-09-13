from cortano import RemoteInterface
from ultralytics import YOLO
import numpy as np
import cv2
import cv2
import numpy as np
import time
import pygame
import sys
from lan import lan

def decode_depth_frame(f):
    R = np.left_shift(frame[:, :, 0].astype(np.uint16), 5)
    G = frame[:, :, 1].astype(np.uint16)
    B = np.left_shift(frame[:, :, 2].astype(np.uint16), 5)
    I = np.bitwise_or(R, G, B)
    return I

if __name__ == "__main__":
    
    model = YOLO("yolov8m.pt")
    
    lan.start("10.0.0.119", 9999, frame_shape=(360, 640))
   
    motor_vals = [0] * 10
    sensor_vals = np.zeros((20,), np.int32)
    
    pygame.init()
    screen = pygame.display.set_mode((640, 360))
    clock = pygame.time.Clock()
    screen.fill((63, 63, 63))
    
    keys_letters = {k[2:]: 0 for k in dir(pygame) if k.startswith("K_")}
    key_number_mappings={
        97: 'a',
        119: 'w',
        115: 's',
        100: 'd',
        1073741906: 'UP',
        1073741905: 'DOWN',
        1073741904: 'LEFT',
        1073741903: 'RIGHT',
        113: 'q'
    }
    
    ball_should_be_within = (240, 115, 70, 70)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:                        
                pygame.quit()
                lan.stop()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                keys_letters[key_number_mappings[event.key]] = 1
            elif event.type == pygame.KEYUP:
                keys_letters[key_number_mappings[event.key]] = 0 
            
        frame = lan.get_frame()
        color_image , depth = frame[:,:640], decode_depth_frame(frame[:,640:])
        disp_img = np.swapaxes(np.flip(color_image, axis=-1), 0, 1)
        surf = pygame.surfarray.make_surface(disp_img)
        
        result = model(color_image, verbose=False)[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        
        for cls, bbox in zip(classes, bboxes):
            if cls in (29,32,47,49,54):
            #if cls in (64, 67, 74): #watch , mouse , phone
                (x, y, x2, y2) = bbox
                width = abs(x-x2)
                height = abs(y-y2)
                pygame.draw.rect(surf, (255,0,0), pygame.Rect(x, y, width, height), 3)
                mid_point = (int(x+width/2), int(y+height/2))
                distance = depth[mid_point[1]][mid_point[0]]/10
                txt = f'{result.names[cls]} - {distance}cm'

        if keys_letters["UP"]: #forward
            motor_vals[0] = -64
            motor_vals[9] = 64
        elif keys_letters["DOWN"]: #reverse
            motor_vals[0] = 64
            motor_vals[9] = -64
        elif keys_letters["LEFT"]: #left
            motor_vals[0] = 60
            motor_vals[9] = 120
        elif keys_letters["RIGHT"]: #right
            motor_vals[0] = -120
            motor_vals[9] = -60
        elif keys_letters["w"]: #up
            motor_vals[6] = 60
        elif keys_letters["s"]: #down
            motor_vals[6] = -60
        elif keys_letters["a"]: #open
            motor_vals[7] = 60
        elif keys_letters["d"]: #close
            motor_vals[7] = -60
        elif keys_letters["q"]:
            pygame.quit()
            lan.stop()
            sys.exit(0)
        else:
            motor_vals = [0] * 10

        lan.send({"motor":motor_vals})
        
        msg = lan.recv()
        if msg and isinstance(msg, dict) and "sensor" in msg:
            sensor_vals = msg["sensor"]
            potentiometer_reading = sensor_vals[0] # At rest its 2512 , at highest its 3638.0
            claw1_switch = int(sensor_vals[1]) #1 or 0
            claw2_switch = int(sensor_vals[2]) # 1 or 0
    
        
        if frame is not None: 
            pygame.draw.rect(surf, (0,255,0), pygame.Rect(*ball_should_be_within), 3)
            screen.blit(surf, (0,0))
            
        pygame.display.flip()
    