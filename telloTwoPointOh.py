import cv2
import pygame
import numpy as np
import time
import os
import tensorflow as tf
import logging
from keras.models import model_from_json
from keras import backend as K

from djitellopy import Tello
from utils import label_map_util
from pygame.locals import *
from simple_pid import PID

# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 20 # --> means an update every 50 ms
# Logging setup
log_file_name = 'logs/blackbox-' + time.strftime('%d_%b_%Y_%H_%M_%S') + '.log'
logging.basicConfig(filename=log_file_name, format='%(asctime)s;%(message)s', level=logging.INFO)
logging.info('timestamp;mode;left_right_vel;for_back_vel;up_down_vel;yaw_vel;target_x;target_y;target_side_len')

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
            - P toggle logging
    """

    def __init__(self):   
        # Object Detection model preparation 
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
        
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        print("Loaded object detection model from disk")

        # Controller model preparation 
        CTR_PATH_TO_CKPT = 'ann_controller/controller_model.pb'  # Path to frozen detection graph. This is the actual model that is used for the object detection.

        # Load a (frozen) Tensorflow model into memory.
        self.controller_graph = tf.Graph()
        with self.controller_graph.as_default():
            ctr_graph_def = tf.GraphDef()
            with tf.gfile.GFile(CTR_PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                ctr_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(ctr_graph_def, name='')
            #op = self.controller_graph.get_operations()
            #[print(m.values()) for m in op]
            #[print(m.name) for m in op]
        
        self.ctr_input = self.controller_graph.get_tensor_by_name('dense_1_input:0')
        self.ctr_output = self.controller_graph.get_tensor_by_name('dense_3/BiasAdd:0')
        print("Loaded controller model from disk")
        
        # Init pygame
        pygame.init()
        self.screen_width = 960
        self.screen_height = 720

        # Create pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        self.basicfont = pygame.font.SysFont(None, 32)

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Logging init
        self.log_to_blackbox = False

        # Drone velocities between -100~100
        self.set_velocities(0, 0, 0, 0) #hover
        self.speed = 10

        # Drone Auto Pilot
        self.target_aquired_tick = 0
        self.seek_speed = 40
        self.seek_tick = 0
        self.target = []
        self.mode = "Manual"    #Seek, Manual, Track
        self.pid_x = PID()
        self.pid_x.tunings = (70, 0, 50)
        self.pid_x.setpoint = 0.5 #scaled self.screen_width / 2
        self.pid_x.output_limits = (-40, 40)
        self.pid_y = PID()
        self.pid_y.tunings = (90, 0, 60) #(60, 0, 40)
        self.pid_y.setpoint = 0.5 #scaled self.screen_height / 2
        self.pid_y.output_limits = (-40, 40)
        self.pid_z = PID()     #depth estimated by length of smallest side of detected object box
        self.pid_z.tunings = (150, 0, 90)
        self.pid_z.setpoint = 0.1 #10% of screen
        self.pid_z.output_limits = (-40, 40)
       
        # Joystick control
        try:
            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        except:
            print("no joystick")

        # Instantiate OCV kalman filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # TODO improve model to include depth: position measurement std = 4

        # Drone HUD
        self.show_flight_data = False
        self.detect_target = False
        self.target_aquired = False

        # Only engage RC once take off command sent
        self.send_rc_control = False

        # create update timer
        self.clock = pygame.time.Clock()

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
            with tf.Session(graph=self.detection_graph) as od_sess:
                with self.controller_graph.as_default():
                    with tf.Session(graph=self.controller_graph) as ctr_sess:
                        while not should_stop:
                            for event in pygame.event.get():
                                if event.type == QUIT:
                                    should_stop = True
                                elif event.type == KEYDOWN:
                                    if event.key == K_ESCAPE:
                                        should_stop = True
                                    else:
                                        self.keydown(event.key)
                                elif event.type == KEYUP:
                                    self.keyup(event.key)
                                elif event.type == pygame.JOYBUTTONDOWN:
                                    self.js_keydown(event.button)
                                if event.type == pygame.JOYAXISMOTION:
                                    self.js_axismotion(event.axis, event.value)

                            if frame_read.stopped:
                                frame_read.stop()
                                break

                            self.screen.fill([0, 0, 0])

                            self.frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
                            self.frame = cv2.flip(self.frame, 1)            

                            if self.detect_target:
                                self.get_target(od_sess)                 
                            
                            self.frame = cv2.circle(self.frame, (int(self.screen_width/2), int(self.screen_height/2)), 30, 100, 4)
                            self.frame = np.rot90(self.frame)
                            self.frame = pygame.surfarray.make_surface(self.frame)
                            self.screen.blit(self.frame, (0, 0))
                            
                            if self.show_flight_data:
                                self.flight_data()

                            pygame.display.update()

                            self.update(ctr_sess)
                            self.clock.tick(FPS)

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def get_target(self, sess):
        frame_small = cv2.resize(self.frame, (int(self.screen_width/2),int(self.screen_height/2)))
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame_small, axis=0)
        # Actual detection.
        (boxes, classes) = sess.run(
            [self.boxes, self.classes],
            feed_dict={self.image_tensor: image_np_expanded})
        try:
            target_index = np.where(classes[0] == 47.)[0][0]  #43 = Tennis Racket, 47 = Cup, 33 = suitcase, 44= bottle
            target_side_len, target_area_coord, target_centroid = self.target_side_len_coord_centroid(
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

            self.target = [target_centroid[0], target_centroid[1], target_side_len]

            control_errors = self.control_errors()
            print(control_errors) #remove
            if abs(control_errors[0]) < 20 and abs(control_errors[1]) < 20 and abs(control_errors[2]) < 20:
                marker_col = (0, 255, 0)
            else:
                marker_col = (200, 0, 0)
            self.frame = cv2.drawMarker(self.frame, (self.target[0], self.target[1]), marker_col, cv2.MARKER_CROSS)
            self.frame = cv2.rectangle(self.frame, target_area_coord[0], target_area_coord[1], marker_col, 2)
        
        if self.target_aquired == True:
            """
            Target previoulsly detected but could be temporarely lost.
            If not lost, update kalmann state with measurement
            If lost, predict position using kalmann filter.
            If lost for more than 10 frames, transition to "Seek" mode
            """
            if target_index != -1:
                measured = np.array([[np.float32(self.target[0])], [np.float32(self.target[1])]])
                self.kf.correct(measured)
                kf_target = self.kf.predict()
                marker_col = (0, 0, 255)
            else:
                self.target_aquired_tick += 1
                kf_target = self.kf.predict()
                
                if kf_target[0,0] < 0:
                    kf_target[0,0] = 0
                if kf_target[0,0] > self.screen_width:
                    kf_target[0,0] = self.screen_width
                if kf_target[1,0] < 0:
                    kf_target[1,0] = 0
                if kf_target[1,0] > self.screen_height:
                    kf_target[1,0] = self.screen_height

                marker_col = (0, 255, 0)
                if self.target_aquired_tick < 5:
                    self.target = [int(kf_target[0,0]),
                                    int(kf_target[1,0]),
                                    int(self.pid_z.setpoint * self.screen_width)] #need to pass predicted area
                else:
                    self.target_aquired = False
                    self.target = []
                    if self.mode == "Track":
                        self.mode = "Seek"
                        self.seek_tick = 0
            self.frame = cv2.drawMarker(self.frame, (kf_target[0,0], kf_target[1,0]), marker_col, cv2.MARKER_SQUARE)

    def target_side_len_coord_centroid(self, x1, x2, y1, y2):
        """
        Returns the centroid of the given coordinates, the length of the smallest side of the detection box
        and the coordinates of the latter
        """
        side_len = None
        coord1 = None
        coord2 = None
        centroid = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
        if (x2 - x1) <= (y2 - y1):
            side_len = int(x2 - x1)
            coord1 = (int(centroid[0] - ((x2 - x1) / 2)), int(centroid[1] - ((x2 - x1) / 2)))
            coord2 = (int(centroid[0] + ((x2 - x1) / 2)), int(centroid[1] + ((x2 - x1) / 2)))
        else:
            side_len = int(y2 - y1)
            coord1 = (int(centroid[0] - ((y2 - y1) / 2)), int(centroid[1] - ((y2 - y1) / 2)))
            coord2 = (int(centroid[0] + ((y2 - y1) / 2)), int(centroid[1] + ((y2 - y1) / 2)))
        return side_len, (coord1, coord2), centroid

    def min_vel(self, vel):
        if -3 < vel < 3:
            vel = 0
        elif 3 <= vel < 7:
            vel = 7
        elif -7 < vel <= -3:
            vel = -7
        return vel
    
    def ANN_control(self, actual, sess):
        # Normalise actuals
        norm_actuals = [actual[0] / (self.screen_width * 1.0),
            actual[2] / (self.screen_width * 1.0),
            actual[1] / (self.screen_height * 1.0)]

        # Actual detection.
        ann_ctr_cmd = sess.run(
            self.ctr_output,
            feed_dict={self.ctr_input: np.array([norm_actuals])})[0]
        print(ann_ctr_cmd) #remove
        
        self.set_velocities(left_right_vel = self.min_vel(int(60 * ann_ctr_cmd[0])),
                            up_down_vel = self.min_vel(int(60 * ann_ctr_cmd[1])),
                            for_back_vel = self.min_vel(int(60 * ann_ctr_cmd[2])),
                            yaw_vel = self.min_vel(int(60 * ann_ctr_cmd[0])))        
    
    def PID_control(self, actual):
        # Normalise actuals
        norm_actuals = [actual[0] / (self.screen_width * 1.0),
            actual[1] / (self.screen_height * 1.0),
            actual[2] / (self.screen_width * 1.0)]

        self.set_velocities(left_right_vel = self.min_vel(int(.6 * self.pid_x(norm_actuals[0]))),
                            up_down_vel = self.min_vel(int(self.pid_y(norm_actuals[1]))),
                            for_back_vel = self.min_vel(int(self.pid_z(norm_actuals[2]))),
                            yaw_vel = self.min_vel(int(.4 * self.pid_x(norm_actuals[0]))))
    
    def control_errors(self):
        return (self.target[0] - (self.pid_x.setpoint * self.screen_width), 
            self.target[1] - (self.pid_y.setpoint * self.screen_height),
            self.target[2] - (self.pid_z.setpoint * self.screen_width))

    def flight_data(self):
        text = "Mode: " + self.mode + ". "

        try:
            text = text + "Battery: " + self.tello.get_battery().rstrip("\r\n") + "%. "
        except:
            text = text + "No battery data! "
        text = text + "Logging "

        if self.log_to_blackbox:
            text = text + "On. "
        else:
            text = text + "Off. "

        try:
            rendered_text = self.basicfont.render(text, True, (10, 10, 10))
            self.screen.blit(rendered_text, (10,680))
        except:
            print("Issue with text e.g. null")
    
    def set_velocities(self, left_right_vel = 999, for_back_vel = 999, up_down_vel = 999, yaw_vel = 999):
        """
        Only function where velocities are set. Only set vel if value is not 999.
        This means can set one DoF without affecting other.
        If called with all zeros values, corresponds to hovering
        """
        if left_right_vel != 999:
            self.left_right_velocity = left_right_vel
        if for_back_vel != 999:
            self.for_back_velocity = for_back_vel
        if up_down_vel != 999:
            self.up_down_velocity = up_down_vel
        if yaw_vel != 999:
            self.yaw_velocity = yaw_vel

    def seek_target(self):
        """
        Executes a hardcoded search pattern:
            1. Clockwise 360 deg at current height
            2. Up for 1 second
            3. back to 1.
        """
        seek_speed = 40
        #self.set_velocities(yaw_vel = seek_speed)
        # Timings for a complete 360 deg at various rotational velocities
        # 20 = 39s
        # 30 = 22s
        # 40 = 14s
        # 50 = 10s

        self.seek_tick += 1
        if self.seek_tick <= 15:
            self.set_velocities(0, 0, 0, 0)
        elif self.seek_tick <= 200:
            self.set_velocities(yaw_vel = seek_speed)
        elif 200 < self.seek_tick < 215:
            self.set_velocities(up_down_vel = seek_speed)
        elif self.seek_tick == 215:
            self.seek_tick = 0

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
        elif key == pygame.K_o:  # toggle show flight data
            self.show_flight_data = not self.show_flight_data
        elif key == pygame.K_u:  # toggle detect object
            self.detect_target = not self.detect_target
            if self.mode != "Manual" and not self.detect_target:
                self.mode = "Manual"
        elif key == pygame.K_p:  # toggle logging and increament session id
            self.log_to_blackbox = not self.log_to_blackbox

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

    def js_keydown(self, button):
        if button == 0:
            self.set_velocities(0, 0, 0, 0) #hover
            if self.mode != "Manual":
                self.mode = "Manual"
            else:
                self.mode = "Seek"
        elif button == 1:
            self.show_flight_data = not self.show_flight_data
        elif button == 4:
            self.log_to_blackbox = not self.log_to_blackbox
        elif button == 5:
            self.detect_target = not self.detect_target
            if self.mode != "Manual" and not self.detect_target:
                self.mode = "Manual"
        elif button == 7:
            self.tello.takeoff()
            self.send_rc_control = True
        elif button == 6:
            self.tello.land()
            self.send_rc_control = False

    def js_axismotion(self, axis, value):
        if axis == 1:
            # up = -1; down = 1
            self.set_velocities(up_down_vel = int(value * -60))
        if axis == 2:
            # cw = -1; ccw = 1
            self.set_velocities(yaw_vel = int(value * -60))
        if axis == 3:
            # for = -1; back = 1
            self.set_velocities(for_back_vel = int(value * -60))
        if axis == 4:
            # left = -1; right = 1
            self.set_velocities(left_right_vel = int(value * 60))

    def update(self, ctr_sess):
        if self.mode == "Seek":
            self.set_velocities(0 ,0, 0, 0) #hover
            self.detect_target = True
            self.seek_target()
        elif self.mode == "Track":
            self.set_velocities(0, 0, 0, 0) #hover
            #self.ANN_control(self.target, ctr_sess)
            self.PID_control(self.target)

        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity,
                                        self.for_back_velocity,
                                        self.up_down_velocity,
                                        self.yaw_velocity)
        if self.log_to_blackbox:
            try:
                logging.info('%s;%s;%s;%s;%s;%s;%s;%s',
                        self.mode,
                        self.left_right_velocity,
                        self.for_back_velocity,
                        self.up_down_velocity,
                        self.yaw_velocity,
                        self.target[0],
                        self.target[1],
                        self.target[2])
            except:
                logging.info('%s;%s;%s;%s;%s',
                        self.mode,
                        self.left_right_velocity,
                        self.for_back_velocity,
                        self.up_down_velocity,
                        self.yaw_velocity)
def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()

if __name__ == '__main__':
    main()