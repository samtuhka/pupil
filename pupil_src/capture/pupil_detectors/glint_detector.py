# make shared modules available across pupil_src
if __name__ == '__main__':
    from sys import path as syspath
    from os import path as ospath
    loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
    syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
    del syspath, ospath

import time
import math
import cv2
from time import sleep
from file_methods import Persistent_Dict
import numpy as np
from methods import *

from glfw import *
from gl_utils import  adjust_gl_view, clear_gl_screen,basic_gl_setup,make_coord_system_norm_based,make_coord_system_pixel_based
from pyglui.cygl.utils import draw_gl_texture

# gui
from pyglui import ui


# logging
import logging
logger = logging.getLogger(__name__)


class Glint_Detector(object):

    def __init__(self, g_pool, customSettings = None):
        super(Glint_Detector, self).__init__()
        self.g_pool = g_pool
        if not customSettings:
            self.session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_glint_detector') )
        else:
            self.session_settings = customSettings

        self.glint_dist = self.session_settings.get('glint_dist', 3.0)
        self.glint_thres = self.session_settings.get('glint_thres', 5)
        self.glint_min = self.session_settings.get('glint_min',50)
        self.glint_max = self.session_settings.get('glint_max',750)
        self.dilate = self.session_settings.get('dilate',0)

        self.prev_glint = (0,0,0)
        self.prev_timestamp = 0
        self.vel_sum = 0.
        self.vel_n = 0.001
        #debug window
        self.suggested_size = 640,480
        self._window = None
        self.window_should_open = False
        self.window_should_close = False


    def bin_thresholding(image, image_lower=0, image_upper=256):
        binary_img = cv2.inRange(image, np.asarray(image_lower),
                np.asarray(image_upper))
        return binary_img


    def irisDetection(self, img, pupil):
        pupilCenter = pupil['center']
        output = img.copy()
        minRad = int(pupil['diameter'])
        maxRad = int(minRad*2.5)
        output = cv2.blur(output,(8,8))
        output = cv2.Canny(output, 10, 20)
        val,output = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(output,cv2.cv.CV_HOUGH_GRADIENT,1,100, param1=50,param2=30,minRadius=minRad,maxRadius=maxRad)
        mean = 0
        meanX = 0
        meanY = 0
        n = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                if (math.sqrt((i[0] - pupilCenter[0])**2 + (i[1] - pupilCenter[1])**2) <  minRad):
                    n += 1
                    mean += i[2]
                    meanX += i[0]
                    meanY += i[1]
                    #cv2.circle(img,(int(i[0]), int(i[1])), i[2],(1,0,0),2)
                    cv2.circle(img,(int(pupilCenter[0]),int(pupilCenter[1])), i[2],(1,0,0),2)
        if n:
            mean = mean/n
            meanX /= n
            meanY /= n
        #cv2.circle(img,(int(pupilCenter[0]),int(pupilCenter[1])), mean,(1,0,0),2)

    def settings(self):
        return self.session_settings

    def update(self):
        self.glint_dist = self.session_settings.get('glint_dist', 3.0)
        self.glint_thres = self.session_settings.get('glint_thres', 5)
        self.glint_min = self.session_settings.get('glint_min',50)
        self.glint_max = self.session_settings.get('glint_max',750)
        self.dilate = self.session_settings.get('dilate',0)


    def filterGlints(self, frame, glints, pupil, eye_id):
        timestamp = frame.timestamp
        pupilDiameter = pupil['diameter']
        minGlint = None
        minDist = 100000000
        secondDist = minDist
        secondGlint = None
        minVel  = 0
        if pupil['confidence']> 0.0:
            pupilCenter = pupil['ellipse']['center']
            maxDist = self.glint_dist * (1.0*pupilDiameter/2)
            for glint in glints:
                dist = math.sqrt((glint[1] - pupilCenter[0])**2 + (glint[2] - pupilCenter[1])**2)
                dt = timestamp - self.prev_timestamp
                vel = math.sqrt((glint[1] - self.prev_glint[1])**2 + (glint[2] - self.prev_glint[2])**2) / dt
                if dist < maxDist and dist < secondDist and (vel < (1.5*self.vel_sum/self.vel_n) or self.vel_n < 5*30):
                    secondDist = minDist
                    minDist = dist
                    secondGlint = minGlint
                    minGlint = glint
                    minVel = vel
        glints = []
        if minGlint:
            glints = [minGlint,[timestamp,0,0,0,0, eye_id]]
            self.prev_glints = minGlint
            self.prev_timestamp = timestamp
            if pupil['confidence']> 0.6:
                self.vel_sum += minVel
                self.vel_n += 1.
        if self.vel_n > 1800:
            self.vel_n = 0.001
            self.vel_sum = 0
        if minGlint and secondGlint:
            min = np.array(minGlint[1:3]) - np.array(list(pupilCenter))
            second = np.array(secondGlint[1:3]) - np.array(list(pupilCenter))
            #angle = math.acos(np.dot(min, second) / ((np.sum(min**2)**0.5) * (np.sum(second**2)**0.5) ))
            c = math.sqrt((minGlint[1] - secondGlint[1])**2 + (minGlint[2] - secondGlint[2])**2)
            if c < 50:
                glints = [minGlint, secondGlint]
        if not glints:
            glints = [[timestamp,0,0,0,0, eye_id], [timestamp,0,0,0,0, eye_id]]
        return glints

    def glint(self,frame, eye_id, u_roi, pupil, roi):

        if self.window_should_open:
            self.open_window((frame.img.shape[1],frame.img.shape[0]))
        if self.window_should_close:
            self.close_window()

        gray = frame.gray[u_roi.view]
        p_r = Roi(gray.shape)
        pupil_img = gray[p_r.view]

        hist = cv2.calcHist([pupil_img],[0],None,[256],[0,256])
        bins = np.arange(hist.shape[0])
        spikes = bins[hist[:,0]>40]

        if spikes.shape[0] >0:
            highest_spike = spikes.max()
        else:
            highest_spike = 255

        spectral_offset = self.glint_thres

        hist *= 1./hist.max()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        spec_mask = bin_thresholding(pupil_img, image_upper=highest_spike - spectral_offset)
        cv2.erode(spec_mask, kernel,spec_mask, iterations=1)

        spec_mask= cv2.morphologyEx(spec_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        spec_mask = cv2.morphologyEx(spec_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        spec_mask = cv2.erode(spec_mask, kernel, iterations=1)
        spec_mask = cv2.morphologyEx(spec_mask, cv2.MORPH_DILATE, kernel, iterations=self.dilate)

        spec_mask[:roi.lY] = 255
        spec_mask[roi.uY:] = 255
        spec_mask[:,:roi.lX] = 255
        spec_mask[:,roi.uX:] = 255

        if self._window:
            img = frame.img
            overlay =  img[u_roi.view][p_r.view]
            b,g,r = overlay[:,:,0],overlay[:,:,1],overlay[:,:,2]
            g[:] = cv2.min(g,spec_mask)

            r_min = int((self.glint_min**0.5) / math.pi)
            r_max = int((self.glint_max**0.5) / math.pi)

            cv2.circle(img,(30,30), r_min,(0,255,0),1)
            cv2.circle(img,(30,30), r_max,(0,0,255),1)

            overlay =  img[u_roi.view][roi.view]
            overlay[::2,0] = 255
            overlay[::2,-1]= 255
            overlay[0,::2] = 255
            overlay[-1,::2]= 255

            if pupil['confidence']> 0.0:
                pupilCenter = pupil['ellipse']['center']
                pupilDiameter = pupil['diameter']
                maxDist = int(self.glint_dist * (1.0*pupilDiameter/2))
                cv2.circle(img,(int(pupilCenter[0]),int(pupilCenter[1])), maxDist,(255,0,0),1)
            self.gl_display_in_window(img)
            overlay = None
        glints = []
        contours, hierarchy = cv2.findContours(spec_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.glint_min and area< self.glint_max:
                centroid = self.contourCenter(cnt)
                newRow = [0, centroid[0], centroid[1], centroid[0]*1.0/frame.img.shape[1], (frame.img.shape[0]-centroid[1]*1.0)/frame.img.shape[0], eye_id]
                glints.append (newRow)

        #if (pupil['confidence']):
        #    self.irisDetection(gray, pupil)
        
        glints = self.filterGlints(frame, glints, pupil, eye_id)
        return glints


    def contourCenter(self, cnt):
        m = cv2.moments(cnt)
        if(m['m00']!=0):
            retVal =  ( int(m['m10']/m['m00']),int(m['m01']/m['m00']))
        else:
            retVal = (-1,-1)
        return retVal


    def init_gui(self,sidebar):
        self.menu = ui.Growing_Menu('Glint Detector')
        self.menu.append(ui.Slider('glint_dist',self,label='Distance from pupil',min=0,max=5,step=0.25))
        self.menu.append(ui.Slider('glint_thres',self,label='Intensity offset',min=0,max=255,step=1))
        self.menu.append(ui.Slider('glint_min',self,label='Min size',min=1,max=250,step=1))
        self.menu.append(ui.Slider('glint_max',self,label='Max size',min=50,max=1000,step=5))
        self.menu.append(ui.Slider('dilate',self,label='Dilate',min=0,max=1,step=1))
        self.menu.append(ui.Button('Open debug window', self.toggle_window))
        sidebar.append(self.menu)

    def toggle_window(self):
        if self._window:
            self.window_should_close = True
        else:
            self.window_should_open = True

    def open_window(self,size):
        if not self._window:
            if 0: #we are not fullscreening
                monitor = glfwGetMonitors()[self.monitor_idx]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= size

            active_window = glfwGetCurrentContext()
            self._window = glfwCreateWindow(height, width, "Glint Detector Debug Window", monitor=monitor, share=active_window)
            if not 0:
                glfwSetWindowPos(self._window,200,0)

            self.on_resize(self._window,height,width)

            #Register callbacks
            glfwSetWindowSizeCallback(self._window,self.on_resize)
            # glfwSetKeyCallback(self._window,self.on_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)

            # gl_state settings
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()

            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)

            self.window_should_open = False

    # window calbacks
    def on_resize(self,window,w,h):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)
        glfwMakeContextCurrent(active_window)

    def on_close(self,window):
        self.window_should_close = True

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False

    def gl_display_in_window(self,img):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)
        clear_gl_screen()
        # gl stuff that will show on your plugin window goes here
        make_coord_system_norm_based()
        draw_gl_texture(img,interpolation=False)
        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)

    def cleanup(self):
        self.session_settings['glint_thres'] = self.glint_thres
        self.session_settings['glint_min'] = self.glint_min
        self.session_settings['glint_max'] = self.glint_max
        self.session_settings['glint_dist'] = self.glint_dist
        self.session_settings['dilate'] = self.dilate
        self.session_settings.close()

