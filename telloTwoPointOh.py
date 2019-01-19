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
        self.set_velocities() #hover
        self.speed = 10

        # Drone Auto Pilot
        self.target_aquired_tick = 0
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
        self.pid_z = PID()     #depth control estimated by size of detect object e.g. size ~= (screen height x width)/20
        self.pid_z.tunings = (.1, 0, 1)
        self.pid_z.setpoint = (self.screen_height * self.screen_width) / 20  #5% of screen
        self.pid_z.output_limits = (-40, 40)

        # Virtual joystick control
        #self.joystick_engaged = False

        # Instantiate OCV kalman filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # TODO improve model to include depth

        # Drone HUD
        self.show_flight_data = False
        self.detect_target = False
        self.target_aquired = False

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
                        # elif event.type == MOUSEBUTTONDOWN:
                        #     self.joystick_engaged = True
                        # elif event.type == MOUSEBUTTONUP:
                        #     self.joystick_engaged = False
                        #     self.set_velocities(0, 0, 0, 0) #hover

                    if frame_read.stopped:
                        frame_read.stop()
                        break

                    self.screen.fill([0, 0, 0])

                    self.frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)

                    if self.show_flight_data:
                        self.flight_data()

                    self.frame = cv2.flip(self.frame, 1)            

                    if self.detect_target:
                        self.get_target(sess)                 
                    
                    self.frame = cv2.circle(self.frame, (int(self.screen_width/2), int(self.screen_height/2)), 30, 100, 4)
                    self.frame = np.rot90(self.frame)
                    self.frame = pygame.surfarray.make_surface(self.frame)
                    self.screen.blit(self.frame, (0, 0))
                    pygame.display.update()

                    time.sleep(1 / FPS) # Could improve taking into account time to execute code...

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def get_target(self, sess):
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
            target_index = np.where(classes[0] == 44.)[0][0]  #43 = Tennis Racket, 47 = Cup, 33 = suitcase, 44= bottle
            target_area, target_area_coord, target_centroid = self.target_area_coord_centroid(
                self.screen_width * boxes[0][target_index][1],
                self.screen_width * boxes[0][target_index][3],
                self.screen_height * boxes[0][target_index][0],
                self.screen_height * boxes[0][target_index][2])
        except:
            target_index = -1
        
        if target_index != -1:
            self.target_aquired = True
            self.target_aquired_tick = 0
            if self.mode == "Seek":
                self.mode = "Track"

            self.target = [target_centroid[0], target_centroid[1], target_area]

            if (abs(self.target[0] - self.pid_x.setpoint) < 20 and 
                abs(self.target[1] - self.pid_y.setpoint) < 20 and
                abs(self.target[2] - self.pid_z.setpoint) < 4000):
                marker_col = (0, 255, 0)
            else:
                marker_col = (200, 0, 0)
            self.frame = cv2.drawMarker(self.frame, (self.target[0], self.target[1]), marker_col, cv2.MARKER_CROSS)
            self.frame = cv2.rectangle(self.frame, target_area_coord[0], target_area_coord[1], marker_col, 2)
        
        if self.target_aquired == True:
            # Target previoulsly detected but could be temporarely lost.
            # If not lost, update kalmann state with measurement
            # If lost, predict position using kalmann filter.
            # If lost for more than 10 frames, transition to "Seek" mode
            if target_index != -1:
                measured = np.array([[np.float32(self.target[0])], [np.float32(self.target[1])]])
                self.kf.correct(measured)
                kf_target = self.kf.predict()
                marker_col = (200, 0, 0)
            else:
                self.target_aquired_tick += 1
                kf_target = self.kf.predict()
                marker_col = (0, 255, 0)
            self.frame = cv2.drawMarker(self.frame, (kf_target[0,0], kf_target[1,0]), marker_col, cv2.MARKER_SQUARE)

            if self.target_aquired_tick < 10:
                self.target = [kf_target[0,0], kf_target[1,0], self.pid_z.setpoint] #need to pass predicted area
            else:
                self.target_aquired = False
                if self.mode == "Track":
                    self.mode = "Seek"
                    print(self.mode)

    def target_area_coord_centroid(self, x1, x2, y1, y2):
        # Returns the centroid of the given coordinates, the area of the square created by the smallest side
        # and the coordinates of the latter
        area = None
        coord1 = None
        coord2 = None
        centroid = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
        if (x2 - x1) <= (y2 - y1):
            area = int((x2 - x1)**2)
            coord1 = (int(centroid[0] - ((x2 - x1) / 2)), int(centroid[1] - ((x2 - x1) / 2)))
            coord2 = (int(centroid[0] + ((x2 - x1) / 2)), int(centroid[1] + ((x2 - x1) / 2)))
        else:
            area = int((y2 - y1)**2)
            coord1 = (int(centroid[0] - ((y2 - y1) / 2)), int(centroid[1] - ((y2 - y1) / 2)))
            coord2 = (int(centroid[0] + ((y2 - y1) / 2)), int(centroid[1] + ((y2 - y1) / 2)))
        return area, (coord1, coord2), centroid

    def PID_control(self, actual):
        self.set_velocities(left_right_vel = int(self.pid_x(actual[0])),
                            up_down_vel = int(self.pid_y(actual[1])),
                            for_back_vel = int(self.pid_z(actual[2])))

    def flight_data(self):
        # Expand to show more data from tello
        font = cv2.FONT_HERSHEY_SIMPLEX
        bot_left_corner_txt = (10,700)
        font_scale = 1
        font_color = (255,255,255)
        line_type = 2

        cv2.putText(self.frame,self.tello.get_battery(), 
            bot_left_corner_txt, 
            font, 
            font_scale,
            font_color,
            line_type,
            cv2.LINE_AA)
    
    def set_velocities(self, left_right_vel = 999, for_back_vel = 999, up_down_vel = 999, yaw_vel = 999):
        # Only function where velocities are set. Only set vel if value is not 999.
        # This means can set one DoF without affecting other.
        # If called with all zeros values, corresponds to hovering
        if left_right_vel != 999:
            self.left_right_velocity = left_right_vel
        if for_back_vel != 999:
            self.for_back_velocity = for_back_vel
        if up_down_vel != 999:
            self.up_down_velocity = up_down_vel
        if yaw_vel != 999:
            self.yaw_velocity = yaw_vel

    # def fly_with_mouse(self):
    #     mouse_pos = pygame.mouse.get_pos()
    #     x_vel = (mouse_pos[0] - (self.screen_width/2))/8    # maxes out at +60 or -60
    #     y_vel = -(mouse_pos[1] - (self.screen_height/2))/6  # maxes out at +60 or -60
    #     self.set_velocities(left_right_vel = int(x_vel), up_down_vel = int(y_vel))

    def seek_target(self):
        seek_tick = 0
        seek_speed = 40
        self.set_velocities(yaw_vel = seek_speed)
        # 20 = 39s
        # 30 = 22s
        # 40 = 14s
        # 50 = 10s

        seek_tick += 1
        if seek_tick == 220:
            self.set_velocities(up_down_vel = seek_speed)
            seek_tick = 0
        if seek_tick == 15:
            self.set_velocities(up_down_vel = 0)

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.set_velocities(for_back_vel = S)
        elif key == pygame.K_DOWN:  # set backward velocity
            self.set_velocities(for_back_vel = -S)
        elif key == pygame.K_LEFT:  # set left velocity
            self.set_velocities(left_right_vel = -S)
        elif key == pygame.K_RIGHT:  # set right velocity
            self.set_velocities(left_right_vel = S)
        elif key == pygame.K_w:  # set up velocity
            self.set_velocities(up_down_vel = S)
        elif key == pygame.K_s:  # set down velocity
            self.set_velocities(up_down_vel = -S)
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.set_velocities(yaw_vel = -S)
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.set_velocities(yaw_vel = S)
        elif key == pygame.K_m:  # toggle auto pilot on
            self.set_velocities(0, 0, 0, 0) #hover
            if self.mode != "Manual":
                self.mode = "Manual"
            else:
                self.mode = "Seek"
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
            self.set_velocities(for_back_vel = 0)
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.set_velocities(left_right_vel = 0)
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.set_velocities(up_down_vel = 0)
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.set_velocities(yaw_vel = 0)
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        # if self.joystick_engaged == True:
        #     self.fly_with_mouse()

        if self.mode == "Seek":
            self.set_velocities(0 ,0, 0, 0) #hover
            self.target = []
            self.seek_target()
        elif self.mode == "Track":
            self.set_velocities(0, 0, 0, 0) #hover
            self.PID_control(self.target)

        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity,
                                        self.for_back_velocity,
                                        self.up_down_velocity,
                                        self.yaw_velocity)

def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()

if __name__ == '__main__':
    main()