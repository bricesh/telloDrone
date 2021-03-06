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

"""
https://github.com/damiafuentes/DJITelloPy
https://github.com/m-lundberg/simple-pid
"""

# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 15

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
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
        self.mode = "Manual"    #Auto, Seek, Manual, Track
        self.pid_x = PID()
        self.pid_x.tunings = (.1, 0, 1) #(.1, 0, 1)
        self.pid_x.setpoint = self.screen_width / 2
        self.pid_x.output_limits = (-40, 40)
        self.pid_y = PID()
        self.pid_y.tunings = (.15, 0.1, .8)
        self.pid_y.setpoint = self.screen_height / 2
        self.pid_y.output_limits = (-40, 40)
        self.joystick_engaged = False

        # Drone HUD
        self.show_flight_data = False

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

                    frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip( frame, 1 )
                    frame_small = cv2.resize(frame, (int(self.screen_width/2),int(self.screen_height/2)))

                    if self.mode == "Seek" or self.mode == "Track":
                        #self.mode = "Seek"
                        #print(self.mode)

                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(frame_small, axis=0)
                        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                        # Each box represents a part of the image where a particular object was detected.
                        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                        # Each score represent how level of confidence for each of the objects.
                        # Score is shown on the result image, together with the class label.
                        #scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                        #num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                        # Actual detection.
                        (boxes, classes) = sess.run(
                            [boxes, classes],
                            feed_dict={image_tensor: image_np_expanded})
                        try:
                            target_index = np.where(classes[0] == 47.)[0][0]  #43 = Tennis Racket, 47 = Cup
                            norm_target_coord = (boxes[0][target_index][1]+(boxes[0][target_index][3]-boxes[0][target_index][1])/2, boxes[0][target_index][0]+(boxes[0][target_index][2]-boxes[0][target_index][0])/2)
                        except:
                            target_index = -1
                        
                        if target_index != -1:
                            print(classes[0][target_index], norm_target_coord)
                            self.mode = "Track"
                            print(self.mode)
                            self.target = [int(self.screen_width * norm_target_coord[0]), int(self.screen_height * norm_target_coord[1])]
                            frame = cv2.drawMarker(frame, tuple(self.target), 200, cv2.MARKER_CROSS)
                    
                    if self.show_flight_data:
                        frame = self.flight_data(frame)
                    frame = cv2.circle(frame, (int(self.screen_width/2), int(self.screen_height/2)), 30, 100, 4)
                    frame = np.rot90(frame)
                    frame = pygame.surfarray.make_surface(frame)
                    self.screen.blit(frame, (0, 0))
                    pygame.display.update()

                    time.sleep(1 / FPS) # Could improve taking into account time to execute code...

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    #def distance(self, a_x, a_y, b_x, b_y):
    #    return ((a_x - b_x)**2 + (a_y - b_y)**2)**.5

    def PID_control(self, set_point, actual):
        x_control_signal_pid = self.pid_x(actual[0])
        y_control_signal_pid = self.pid_y(actual[1])
        self.left_right_velocity = int(x_control_signal_pid)
        #print("X control command", x_control_signal_pid)
        self.up_down_velocity = int(y_control_signal_pid)
        #print("Y control command", y_control_signal_pid)

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
        x_vel = (mouse_pos[0] - (self.screen_width/2))/8
        self.left_right_velocity = int(x_vel)
        y_vel = -(mouse_pos[1] - (self.screen_height/2))/6
        self.up_down_velocity = int(y_vel)
        print(x_vel,y_vel)

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
                self.mode = "Auto"
                self.tick = 0
            print(self.mode)
        elif key == pygame.K_o:  # toggle show flight data
            if self.show_flight_data:
                self.show_flight_data = False
            else:
                self.show_flight_data = True
        elif key == pygame.K_u:  # set yaw counter clockwise velocity
            self.pid_x.Kp += .1
            print(self.pid_x.Kp)

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
            self.fly_with_mouse()

        if self.mode == "Auto":
            self.hover()
            self.mode = "Seek"
            print(self.mode)
        elif self.mode == "Seek":
            self.hover()
            self.target = []
            self.seek_target()
        elif self.mode == "Track":
            self.hover()
            self.PID_control([self.screen_width/2, self.screen_height/2], self.target)

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