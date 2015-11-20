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

from c_methods import eye_filter
from glfw import *
from template import Pupil_Detector

# gui
from pyglui import ui


# logging
import logging
logger = logging.getLogger(__name__)

class Glint_Detector(object):

    def __init__(self, g_pool):
        super(Glint_Detector, self).__init__()
        self.g_pool = g_pool
        self.session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_glint_detector') )

        self.glint_dist = self.session_settings.get('glint_dist', 3.0)
        self.glint_thres = self.session_settings.get('glint_thres',185.)
        self.glint_min = self.session_settings.get('glint_min',0.)
        self.glint_max = self.session_settings.get('glint_max',200.)



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


    def filterGlints(self, frame, glints, pupil, eye_id):
        timestamp = frame.timestamp
        pupilDiameter = pupil['diameter']
        minGlint = None
        minDist = 10000
        secondDist = minDist
        secondGlint = None
        if pupil['confidence']> 0.0:
            pupilCenter = pupil['center']
            maxDist = self.glint_dist * (1.0*pupilDiameter/2)
            for glint in glints:
                dist = math.sqrt((glint[1] - pupilCenter[0])**2 + (glint[2] - pupilCenter[1])**2)
                if dist < maxDist and dist < secondDist:
                    secondDist = minDist
                    minDist = dist
                    secondGlint = minGlint
                    minGlint = glint
        glints = []
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

    def glint(self,frame, eye_id, u_roi, pupil):
        gray = frame.gray[u_roi.view]
        val,binImg = cv2.threshold(gray, self.glint_thres, 255, cv2.THRESH_BINARY)
        timestamp = frame.timestamp
        st7 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
        binImg= cv2.morphologyEx(binImg, cv2.MORPH_OPEN, st7)
        binImg = cv2.morphologyEx(binImg, cv2.MORPH_DILATE, st7, iterations=1)
        binImg = cv2.erode(binImg, st7, iterations=1)
        contours, hierarchy = cv2.findContours(binImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(binImg,contours,-1,(0,255,0),3)
        glints = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.glint_min and area< self.glint_max:
                centroid = self.contourCenter(cnt)
                newRow = [timestamp, centroid[0], centroid[1], centroid[0]*1.0/frame.img.shape[1], (frame.img.shape[0]-centroid[1]*1.0)/frame.img.shape[0], eye_id]
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
        self.menu.append(ui.Slider('glint_thres',self,label='Intensity threshold',min=0,max=255,step=5))
        self.menu.append(ui.Slider('glint_min',self,label='Min size',min=1,max=100,step=1))
        self.menu.append(ui.Slider('glint_max',self,label='Max size',min=50,max=1000,step=5))
        sidebar.append(self.menu)


    def cleanup(self):
        self.session_settings['glint_thres'] = self.glint_thres
        self.session_settings['glint_min'] = self.glint_min
        self.session_settings['glint_max'] = self.glint_max
        self.session_settings['glint_dist'] = self.glint_dist
        self.session_settings.close()

