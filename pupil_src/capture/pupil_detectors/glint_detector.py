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
from gl_utils import  draw_gl_texture,adjust_gl_view, clear_gl_screen, draw_gl_point_norm, draw_gl_polyline,basic_gl_setup,make_coord_system_norm_based,make_coord_system_pixel_based
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


    def filterGlints(self, frame, glints, pupil):
        timestamp = frame.timestamp
        pupilDiameter = pupil['diameter']
        minGlint = None
        minDist = 10000
        if pupil['confidence']>0.60:
            pupilCenter = pupil['center']
            maxDist = 1.5 * pupilDiameter
            for glint in glints:
                dist = math.sqrt((glint[1] - pupilCenter[0])**2 + (glint[2] - pupilCenter[1])**2)
                if dist < maxDist and dist < minDist:
                    minDist = dist
                    minGlint = glint
        if minGlint:
            glints = [minGlint]
        else:
            glints = [[timestamp,0,0,0,0]]
        return glints

    def glint(self,frame, u_roi, pupil):
        gray = frame.gray[u_roi.view]
        val,binImg = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
        timestamp = frame.timestamp

        st7 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
        binImg= cv2.morphologyEx(binImg, cv2.MORPH_OPEN, st7)
        binImg = cv2.morphologyEx(binImg, cv2.MORPH_DILATE, st7, iterations=2)

        contours, hierarchy = cv2.findContours(binImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        glints = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50 and area<500:
                centroid = self.contourCenter(cnt)
                newRow = [timestamp, centroid[0], centroid[1], centroid[0]*1.0/frame.img.shape[1], (frame.img.shape[0]-centroid[1]*1.0)/frame.img.shape[0]]
                glints.append (newRow)
        #if (pupil['confidence']):
        #    self.irisDetection(gray, pupil)
        glints = self.filterGlints(frame, glints, pupil)
        return glints

    def contourCenter(self, cnt):
        m = cv2.moments(cnt)
        if(m['m00']!=0):
            retVal =  ( int(m['m10']/m['m00']),int(m['m01']/m['m00']))
        else:
            retVal = (-1,-1)
        return retVal




