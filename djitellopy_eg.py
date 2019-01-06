from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
from simple_pid import PID

"""
https://github.com/damiafuentes/DJITelloPy
https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
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
        self.mode = "Manual"
        self.pid_x = PID()
        self.pid_x.tunings = (.12, 0, .9) #(.1, 0, 1)
        self.pid_x.setpoint = self.screen_width / 2
        self.pid_x.output_limits = (-40, 40)
        self.pid_y = PID()
        self.pid_y.tunings = (.15, 0.1, .8)
        self.pid_y.setpoint = self.screen_height / 2
        self.pid_y.output_limits = (-40, 40)

        # Drone HUD
        self.show_flight_data = False

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

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

            if frame_read.stopped:
                frame_read.stop()
                break

            self.screen.fill([0, 0, 0])

            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip( frame, 1 )

            if self.mode == "Seek" or self.mode == "Track":
                self.mode = "Seek"
                frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frameGray = cv2.GaussianBlur(frameGray, (5, 5), 0)
                croped_threshold = []
                #_, threshold = cv2.threshold(frameGray, 60, 255, cv2.THRESH_BINARY)
                threshold = cv2.adaptiveThreshold(frameGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
                _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

                    if len(approx) == 4:
                        if self.distance(approx.ravel()[0], approx.ravel()[2], approx.ravel()[1], approx.ravel()[3]) > 100:
                            max_x = max(approx.ravel().reshape(4,2)[:,0])
                            min_x = min(approx.ravel().reshape(4,2)[:,0])
                            max_y = max(approx.ravel().reshape(4,2)[:,1])
                            min_y = min(approx.ravel().reshape(4,2)[:,1])
                            croped_threshold = threshold[min_y:max_y,min_x:max_x]
                            _, croped_contours, _ = cv2.findContours(croped_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            for croped_cnt in croped_contours:
                                croped_approx = cv2.approxPolyDP(croped_cnt, 0.01*cv2.arcLength(cnt, True), True)
                                if len(croped_approx) == 6:
                                    cnt_area = cv2.contourArea(cnt)
                                    cropped_cnt_area = cv2.contourArea(croped_cnt)
                                    if cropped_cnt_area == 0:
                                        cnt_to_cropped_cnt_ratio = 1000
                                    else:
                                        cnt_to_cropped_cnt_ratio = cnt_area / cropped_cnt_area
                                    if cnt_to_cropped_cnt_ratio >= 3.5 and cnt_to_cropped_cnt_ratio <= 4.5:
                                        cv2.drawContours(frame, [approx], 0, (0), 5)
                                        self.mode = "Track"
                                        self.target = [int(min_x + (max_x-min_x)/2), int(min_y + (max_y-min_y)/2)]
                                        frame = cv2.drawMarker(frame, tuple(self.target), 200, cv2.MARKER_CROSS)
                                else:
                                    croped_threshold = []
            
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

    def distance(self, a_x, a_y, b_x, b_y):
        return ((a_x - b_x)**2 + (a_y - b_y)**2)**.5

    def PID_control(self, set_point, actual):
        P_gain_x = .1
        I_gain_x = 0
        D_gain_x = 0
        P_gain_y = .1
        I_gain_y = 0
        D_gain_y = 0
        distance_to_target = self.distance(set_point[0], set_point[1], actual[0], actual[1])
        x_error = set_point[0] - actual[0]
        y_error = set_point[1] - actual[1]
        x_control_signal = int(np.clip(P_gain_x * abs(x_error), 10, 30)) * (int(x_error > 0) - int(x_error < 0))
        y_control_signal = int(np.clip(P_gain_y * abs(y_error), 10, 30)) * (int(y_error > 0) - int(y_error < 0))
        x_control_signal_pid = self.pid_x(actual[0])
        y_control_signal_pid = self.pid_y(actual[1])
        #self.left_right_velocity = x_control_signal
        self.left_right_velocity = int(x_control_signal_pid)
        print("X", x_control_signal, x_control_signal_pid)
        #self.up_down_velocity = y_control_signal
        self.up_down_velocity = int(y_control_signal_pid)
        print("Y", y_control_signal, y_control_signal_pid)
        return [x_control_signal_pid, y_control_signal_pid, distance_to_target]

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
        if self.mode == "Auto":
            self.hover()
            self.mode = "Seek"
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