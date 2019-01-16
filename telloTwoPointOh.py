import cv2
import pygame
import numpy as np
import time
import os
import tensorflow as tf

from djitellopy import Tello
from utils import label_map_util
from pygame.locals import *
from simple_pid import PID

# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 20

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
            - M toggle between Manual and 'Auto Pilot' modes
            - O display flight data
            - U toggle target detection
    """

    def __init__(self):
        # Model preparation 
        MODEL_NAME = 'ssd_mobilenet_v1_coco'
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'  # Path to frozen detection graph. This is the actual model that is used for the object detection.

        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Init pygame
        pygame.init()
        self.screen_width = 960
        self.screen_height = 720

        # Create pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        # Drone Auto Pilot
        self.tick = 0
        self.seek_speed = 40
        self.target = []
        self.mode = "Manual"    #Seek, Manual, Track
        self.pid_x = PID()
        self.pid_x.tunings = (.1, 0, 1) #(.1, 0, 1)
        self.pid_x.setpoint = self.screen_width / 2
        self.pid_x.output_limits = (-40, 40)
        self.pid_y = PID()
        self.pid_y.tunings = (.15, 0.1, .8)
        self.pid_y.setpoint = self.screen_height / 2
        self.pid_y.output_limits = (-40, 40)
        #self.pid_z = PID()     #depth control estimated by size of detect object e.g. size ~= (screen height x width)/4
        #self.pid_z.tunings = (.1, 0, 0)
        #self.pid_z.setpoint = (self.screen_height * self.screen_width) / 4
        #self.pid_z.output_limits = (-40, 40)

        # Virtual joystick control
        self.joystick_engaged = False

        # Instantiate OCV kalman filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

        # Drone HUD
        self.show_flight_data = False
        self.detect_target = False
        self.target_detected = False

        # Only engage RC once take off command sent
        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, int(1000/(FPS + 1)))

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()

        should_stop = False
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                while not should_stop:

                    for event in pygame.event.get():
                        if event.type == USEREVENT + 1:
                            self.update()
                        elif event.type == QUIT:
                            should_stop = True
                        elif event.type == KEYDOWN:
                            if event.key == K_ESCAPE:
                                should_stop = True
                            else:
                                self.keydown(event.key)
                        elif event.type == KEYUP:
                            self.keyup(event.key)
                        elif event.type == MOUSEBUTTONDOWN:
                            self.joystick_engaged = True
                        elif event.type == MOUSEBUTTONUP:
                            self.joystick_engaged = False
                            self.hover()

                    if frame_read.stopped:
                        frame_read.stop()
                        break

                    self.screen.fill([0, 0, 0])

                    self.frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
                    self.frame = cv2.flip(self.frame, 1)            

                    if self.detect_target:
                        self.detectTarget(sess)
                    
                    if self.show_flight_data:
                        self.frame = self.flight_data(self.frame)
                    
                    self.frame = cv2.circle(self.frame, (int(self.screen_width/2), int(self.screen_height/2)), 30, 100, 4)
                    self.frame = np.rot90(self.frame)
                    self.frame = pygame.surfarray.make_surface(self.frame)
                    self.screen.blit(self.frame, (0, 0))
                    pygame.display.update()

                    time.sleep(1 / FPS) # Could improve taking into account time to execute code...

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    #def distance(self, a_x, a_y, b_x, b_y):
    #    return ((a_x - b_x)**2 + (a_y - b_y)**2)**.5

    def detectTarget(self, sess):
        frame_small = cv2.resize(self.frame, (int(self.screen_width/2),int(self.screen_height/2)))
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame_small, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        # Actual detection.
        (boxes, classes) = sess.run(
            [boxes, classes],
            feed_dict={image_tensor: image_np_expanded})
        try:
            target_index = np.where(classes[0] == 47.)[0][0]  #43 = Tennis Racket, 47 = Cup
            norm_target_coord = (boxes[0][target_index][1]+(boxes[0][target_index][3]-boxes[0][target_index][1])/2,
                                    boxes[0][target_index][0]+(boxes[0][target_index][2]-boxes[0][target_index][0])/2)
        except:
            target_index = -1
        
        if target_index != -1:
            self.target_detected = True
            if self.mode == "Seek":
                self.mode = "Track"

            self.target = [int(self.screen_width * norm_target_coord[0]),
                            int(self.screen_height * norm_target_coord[1])]
            self.frame = cv2.drawMarker(self.frame, tuple(self.target), 200, cv2.MARKER_CROSS)
        
        if self.target_detected == True:
            if target_index != -1:
                measured = np.array([[np.float32(self.target[0])], [np.float32(self.target[1])]])
                self.kf.correct(measured)
                kf_target = self.kf.predict()
            else:
                kf_target = self.kf.predict()
            self.frame = cv2.drawMarker(self.frame, (kf_target[0,0], kf_target[1,0]), 300, cv2.MARKER_SQUARE)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        return self.kf.predict()

    def PID_control(self, set_point, actual):
        return (int(self.pid_x(actual[0])), int(self.pid_y(actual[1])))

    def flight_data(self, img):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,700)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        img = cv2.flip( img, 1 )
        cv2.putText(img,self.tello.get_battery(), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType,
            cv2.LINE_AA)
        return cv2.flip( img, 1 )
    
    def hover(self):
        self.left_right_velocity = 0
        self.for_back_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

    def fly_with_mouse(self):
        mouse_pos = pygame.mouse.get_pos()
        x_vel = (mouse_pos[0] - (self.screen_width/2))/8    # maxes out at +60 or -60
        y_vel = -(mouse_pos[1] - (self.screen_height/2))/6  # maxes out at +60 or -60
        return (int(x_vel), int(y_vel))

    def seek_target(self):
        self.yaw_velocity = self.seek_speed
        # 20 = 39s
        # 30 = 22s
        # 40 = 14s
        # 50 = 10s

        self.tick += 1
        if self.tick == 220:
            self.up_down_velocity = self.seek_speed
            self.tick = 0
        if self.tick == 15:
            self.up_down_velocity = 0

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_m:  # toggle auto pilot on
            self.hover()
            if self.mode != "Manual":
                self.mode = "Manual"
            else:
                self.mode = "Seek"
                self.tick = 0
            print(self.mode)
        elif key == pygame.K_o:  # toggle show flight data
            self.show_flight_data = not self.show_flight_data
        elif key == pygame.K_u:  # toggle detect object
            self.detect_target = not self.detect_target

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        if self.joystick_engaged == True:
            (self.left_right_velocity, self.up_down_velocity) = self.fly_with_mouse()

        if self.mode == "Seek":
            self.hover()
            self.target = []
            self.seek_target()
        elif self.mode == "Track":
            self.hover()
            (self.left_right_velocity, self.up_down_velocity) = self.PID_control([self.screen_width/2, self.screen_height/2],
                                                                                    self.target)

        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()

if __name__ == '__main__':
    main()