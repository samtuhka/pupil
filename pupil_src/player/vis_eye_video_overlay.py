'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import sys, os, platform
from glob import glob
import cv2
import math
import numpy as np
from file_methods import Persistent_Dict
from pyglui import ui
from player_methods import transparent_image_overlay
from plugin import Plugin
import av
import copy
from time import sleep
import _thread
from copy import copy
import json
#import pyelse


# helpers/utils
from version_utils import VersionFormat

#capture
from video_capture import EndofVideoFileError,FileSeekError,FileCaptureError,File_Source

#mouse
from glfw import glfwGetCursorPos,glfwGetWindowSize,glfwGetCurrentContext
from methods import normalize,denormalize
from file_methods import Persistent_Dict,save_object

from pupil_detectors import Detector_2D, Detector_3D, Glint_Detector
from ui_roi import UIRoi
import ui_roi


#logging
import logging
logger = logging.getLogger(__name__)

from multiprocessing import Process, Pipe, Queue, Value,active_children, freeze_support


class Global_Container(object):
    pass


def get_past_timestamp(idx,timestamps):
    """
    recursive function to find the most recent valid timestamp in the past
    """
    if idx == 0:
        # if at the beginning, we can't go back in time.
        return get_future_timestamp(idx,timestamps)
    if timestamps[idx]:
        res = timestamps[idx][-1]
        return res
    else:
        return get_past_timestamp(idx-1,timestamps)

def get_future_timestamp(idx,timestamps):
    """
    recursive function to find most recent valid timestamp in the future
    """
    if idx == len(timestamps)-1:
        # if at the end, we can't go further into the future.
        return get_past_timestamp(idx,timestamps)
    elif timestamps[idx]:
        return timestamps[idx][0]
    else:
        idx = min(len(timestamps),idx+1)
        return get_future_timestamp(idx,timestamps)

def get_nearest_timestamp(past_timestamp,future_timestamp,world_timestamp):
    dt_past = abs(past_timestamp-world_timestamp)
    dt_future = abs(future_timestamp-world_timestamp) # abs prob not necessary here, but just for sanity
    if dt_past < dt_future:
        return past_timestamp
    else:
        return future_timestamp

def correlate_eye_world(eye_timestamps,world_timestamps):
    """
    This function takes a list of eye timestamps and world timestamps
    and correlates one eye frame per world frame
    Returns a mapping that correlates a single eye frame index with each world frame index.
    Up and downsampling is used to achieve this mapping.
    """
    # return framewise mapping as a list
    e_ts = eye_timestamps
    w_ts = list(world_timestamps)
    eye_frames_by_timestamp = dict(zip(e_ts,range(len(e_ts))))

    eye_timestamps_by_world_index = [[] for i in world_timestamps]

    frame_idx = 0
    try:
        current_e_ts = e_ts.pop(0)
    except:
        logger.warning("No eye timestamps found.")
        return eye_timestamps_by_world_index

    while e_ts:
        # if the current eye timestamp is before the mean of the current world frame timestamp and the next worldframe timestamp
        try:
            t_between_frames = ( w_ts[frame_idx]+w_ts[frame_idx+1] ) / 2.
        except IndexError:
            break
        if current_e_ts <= t_between_frames:
            eye_timestamps_by_world_index[frame_idx].append(current_e_ts)
            current_e_ts = e_ts.pop(0)
        else:
            frame_idx+=1

    idx = 0
    eye_world_frame_map = []
    # some entiries in the `eye_timestamps_by_world_index` might be empty -- no correlated eye timestamp
    # so we will either show the previous frame or next frame - whichever is temporally closest
    for candidate,world_ts in zip(eye_timestamps_by_world_index,w_ts):
        # if there is no candidate, then assign it to the closest timestamp
        if not candidate:
            # get most recent timestamp, either in the past or future
            e_past_ts = get_past_timestamp(idx,eye_timestamps_by_world_index)
            e_future_ts = get_future_timestamp(idx,eye_timestamps_by_world_index)
            eye_world_frame_map.append(eye_frames_by_timestamp[get_nearest_timestamp(e_past_ts,e_future_ts,world_ts)])
        else:
            # TODO - if there is a list of len > 1 - then we should check which is the temporally closest timestamp
            eye_world_frame_map.append(eye_frames_by_timestamp[eye_timestamps_by_world_index[idx][-1]])
        idx += 1

    return eye_world_frame_map



class Vis_Eye_Video_Overlay(Plugin):
    """docstring This plugin allows the user to overlay the eye recording on the recording of his field of vision
        Features: flip video across horiz/vert axes, click and drag around interface, scale video size from 20% to 100%,
        show only 1 or 2 or both eyes
        features updated by Andrew June 2015
    """
    def __init__(self,g_pool,alpha=0.6,eye_scale_factor=.5,move_around=0,mirror={'0':False,'1':False}, flip={'0':False,'1':False},pos=[(640,10),(10,10)], show_ellipses=True):
        super().__init__(g_pool)
        self.order = .6
        self.menu = None

        # user controls
        self.alpha = alpha #opacity level of eyes
        self.eye_scale_factor = eye_scale_factor #scale
        self.showeyes = 0,1 #modes: any text containg both means both eye is present, on 'only eye1' if only one eye recording
        self.move_around = move_around #boolean whether allow to move clip around screen or not
        self.video_size = [0,0] #video_size of recording (bc scaling)
        self.show_ellipses = show_ellipses

        self.detect_3D = 0
        self.algorithm = 0

        self.persistent_settings_0 = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_eye0') )
        self.persistent_settings_1 = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_eye1') )
        self.persistent_settings_roi = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_roi') )

        self.current_settings_0 = {}
        self.current_settings_1 = {}

        self.gPool0 = Global_Container()
        self.gPool1 = Global_Container()

        self.gPool0.pupil_queue = Queue()
        self.gPool1.pupil_queue = Queue()

        self.center = [0,0]
        self.mouse_img = [[0,0],[0,0]]

        self.g_pool = g_pool

        #below is a fckng mess
        self.pupil_size_min = 40
        self.pupil_size_max = 150
        self.intensity_range = 17
        self.model_sensitivity = 0.997
        self.ellipse_roundness_ratio = 0.1
        self.coarse_filter_min = 150
        self.coarse_filter_max = 300
        self.initial_ellipse_fit_treshhold = 1.8
        self.canny_treshold = 200
        self.canny_ration = 3

        self.pupil_size_min1 = 40
        self.pupil_size_max1 = 150
        self.intensity_range1 = 17
        self.model_sensitivity1 = 0.997
        self.ellipse_roundness_ratio1 = 0.1
        self.coarse_filter_min1 = 150
        self.coarse_filter_max1 = 300
        self.initial_ellipse_fit_treshhold1 = 1.8
        self.canny_treshold1 = 200
        self.canny_ration1 = 3

        uroi0 = self.persistent_settings_roi.get('roi0', False)
        uroi1 = self.persistent_settings_roi.get('roi1', False)
        self.u_r = [UIRoi((480, 640)), UIRoi((480, 640))]


        if uroi0 != False:
            self.u_r[0].lX = uroi0[0]
            self.u_r[0].lY = uroi0[1]
            self.u_r[0].uX = uroi0[2]
            self.u_r[0].uY = uroi0[3]
        if uroi1  != False:
            self.u_r[1].lX = uroi1[0]
            self.u_r[1].lY = uroi1[1]
            self.u_r[1].uX = uroi1[2]
            self.u_r[1].uY = uroi1[3]
            
        self.urActive = False

        self.rec_dir = g_pool.rec_dir

        #variables specific to each eye
        self.eye_frames = []
        self.eye_world_frame_map = []
        self.eye_cap = []
        self.mirror = mirror #do we horiz flip first eye
        self.flip = flip #do we vert flip first eye
        self.pos = [list(pos[0]),list(pos[1])] #positions of 2 eyes
        self.drag_offset = [None,None]

        pupil_detector_eye0 = Detector_2D(g_pool = self.gPool0)
        pupil_detector_eye1 = Detector_2D(g_pool = self.gPool1)

        pupil_detector_eye0_3D = Detector_3D(g_pool = self.gPool0)
        pupil_detector_eye1_3D = Detector_3D(g_pool = self.gPool1)


        self.pupil_detectors2D = [pupil_detector_eye0,pupil_detector_eye1]
        self.pupil_detectors3D = [pupil_detector_eye0_3D,pupil_detector_eye1_3D]

        self.pupil_detectors = self.pupil_detectors2D

        self.glint_settings = {}

        self.glint_dist = 3.0
        self.glint_thres = 5
        self.glint_min = 50
        self.glint_max = 750
        self.dilate = 0
        self.erode = 0

        self.glint_dist1 = 3.0
        self.glint_thres1 = 5
        self.glint_min1 = 50
        self.glint_max1 = 750
        self.dilate1 = 0
        self.erode1 = 0
        self.recalculating = 0
        self.pupil_settings = 'default'

        self.msg = ""

        glint_detector0 = Glint_Detector(g_pool, self.glint_settings)
        glint_detector1 = Glint_Detector(g_pool, self.glint_settings)
        self.glint_detectors = [glint_detector0, glint_detector1]

        # load eye videos and eye timestamps
        if VersionFormat(self.g_pool.meta_info['Capture Software Version'][1:]) < VersionFormat('0.4'):
            eye_video_path = os.path.join(g_pool.rec_dir,'eye.avi'),'None'
            self.eye_timestamps_path = os.path.join(g_pool.rec_dir,'eye_timestamps.npy'),'None'
        else:
            eye_video_path = os.path.join(g_pool.rec_dir,'eye0.*'),os.path.join(g_pool.rec_dir,'eye1.*')
            self.eye_timestamps_path = os.path.join(g_pool.rec_dir,'eye0_timestamps.npy'),os.path.join(g_pool.rec_dir,'eye1_timestamps.npy')

        #try to load eye video and ts for each eye.
        self.eye_ts = []
        for video,ts in zip(eye_video_path,self.eye_timestamps_path):
            try:
                self.eye_cap.append(File_Source(self.g_pool,source_path=glob(video)[0],timestamps=np.load(ts)))
            except(IndexError,FileCaptureError):
                pass
            else:
                self.eye_frames.append(self.eye_cap[-1].get_frame())
            try:
                eye_timestamps = list(np.load(ts))
            except:
                pass
            else:
                self.eye_world_frame_map.append(correlate_eye_world(eye_timestamps,g_pool.timestamps))

        if len(self.eye_cap) == 2:
            logger.debug("Loaded binocular eye video data.")
        elif len(self.eye_cap) == 1:
            logger.debug("Loaded monocular eye video data")
            self.showeyes = (0,)
        else:
            logger.error("Could not load eye video.")
            self.alive = False
            return

        self.setKeys(self.persistent_settings_0, 0)
        self.setKeys(self.persistent_settings_1, 1)

    def unset_alive(self):
        self.alive = False

    def setKeys(self, pupil_settings_new, eye = 0):
        keys = list(self.__dict__)
        for key in keys:
            try:
                if eye == 0:
                    self.__dict__[key] = pupil_settings_new[key]
                else:
                    self.__dict__[key + "1"] = pupil_settings_new[key]
                logger.info("set key with the name '%s'" %key)
            except:
                pass
                #logger.info("no key with the name '%s' in settings" %key)

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Eye Video Overlay')
        self.update_gui()
        self.g_pool.gui.append(self.menu)

    def set_pupil_settings(self, new_settings):
            self.pupil_settings = new_settings
            if not new_settings == "default":
                try:
                    path = os.path.join(self.g_pool.user_dir,'pupil_settings_' + new_settings + '.json')
                    with open(path, 'r') as fp:
                        json_str = fp.read()
                        pupil_settings_new = json.loads(json_str)
                except:
                    logger.error("Settings don't exist")

                self.setKeys(pupil_settings_new, 0)
                self.setKeys(pupil_settings_new, 1)

    def save_settings(self):
        settings_eye0 = Persistent_Dict(os.path.join(self.rec_dir,'user_settings_eye0') )
        settings_eye1 = Persistent_Dict(os.path.join(self.rec_dir,'user_settings_eye1') )
        settings_roi = Persistent_Dict(os.path.join(self.rec_dir,'user_settings_roi') )
        settings_eye0.update(self.current_settings_0)
        settings_eye1.update(self.current_settings_1)
        settings_roi['roi0'] = self.u_r[0].get()
        settings_roi['roi1'] = self.u_r[1].get()

        settings_eye0.close()
        settings_eye1.close()
        settings_roi.close()

        


    def update_gui(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.unset_alive))

        self.menu.append(ui.Switch('detect_3D',self,label="3D detection"))
        self.menu.append(ui.Switch('algorithm',self,label="Algorithm view"))
        self.menu.append(ui.Button("Save settings", self.save_settings))
        self.menu.append(ui.Switch('show_ellipses', self, label="Visualize Recorded Gaze"))
        self.menu.append(ui.Selector('pupil_settings',self,setter=self.set_pupil_settings,selection=['default','indoors','outdoors_sunny', 'outdoors_cloudy', 'vanilla'], labels=['Default', 'Indoors', 'Outdoors Sunny', 'Outdoors Cloudy', 'Vanilla'], label="Pupil settings") )

        pupil0_menu = ui.Growing_Menu('Pupil0')
        pupil0_menu.collapsed = True
        pupil0_menu.append(ui.Slider('pupil_size_min',self,min=0,step=1,max=250,label='Pupil min size'))
        pupil0_menu.append(ui.Slider('pupil_size_max' ,self,min=0,step=1,max=400,label='Pupil max size'))
        pupil0_menu.append(ui.Slider('intensity_range',self,min=0,step=1,max=60,label='Pupil intensity range'))
        pupil0_menu.append(ui.Slider('model_sensitivity',self,min=0.0,step=0.0001,max=1.0,label='Model sensitivity'))
        pupil0_menu[-1].display_format = '%0.4f'
        pupil0_menu.append(ui.Slider('ellipse_roundness_ratio',self,min=0.01,step=0.01,max=1.0,label='ellipse_roundness_ratio'))
        pupil0_menu.append(ui.Slider('coarse_filter_min',self,min=10,step=1,max=500,label='coarse_filter_min'))
        pupil0_menu.append(ui.Slider('coarse_filter_max',self,min=100,step=1,max=1000,label='coarse_filter_max'))
        pupil0_menu.append(ui.Slider('canny_treshold',self,min=50,step=1,max=500,label='canny_treshold'))
        pupil0_menu.append(ui.Slider('canny_ration',self,min=1,step=1,max=20,label='canny_ration'))


        pupil0_menu.append(ui.Button('Reset 3D model', self.reset_3D_Model_eye0 ))

        self.menu.append(pupil0_menu)

        pupil1_menu = ui.Growing_Menu('Pupil1')
        pupil1_menu.collapsed = True
        pupil1_menu.append(ui.Slider('pupil_size_min1',self,min=0,step=1,max=250,label='Pupil min size'))
        pupil1_menu.append(ui.Slider('pupil_size_max1' ,self,min=0,step=1,max=400,label='Pupil max size'))
        pupil1_menu.append(ui.Slider('intensity_range1',self,min=0,step=1,max=60,label='Pupil intensity range'))
        pupil1_menu.append(ui.Slider('model_sensitivity1',self,min=0.0, step=0.0001, max=1.0, label='Model sensitivity'))
        pupil1_menu[-1].display_format = '%0.4f'
        pupil1_menu.append(ui.Slider('ellipse_roundness_ratio1',self,min=0.01,step=0.01,max=1.0,label='ellipse_roundness_ratio'))
        pupil1_menu.append(ui.Slider('coarse_filter_min1',self,min=10,step=1,max=500,label='coarse_filter_min'))
        pupil1_menu.append(ui.Slider('coarse_filter_max1',self,min=100,step=1,max=1000,label='coarse_filter_max'))
        pupil1_menu.append(ui.Slider('canny_treshold1',self,min=50,step=1,max=500,label='canny_treshold'))
        pupil1_menu.append(ui.Slider('canny_ration1',self,min=1,step=1,max=20,label='canny_ration'))
        pupil1_menu.append(ui.Button('Reset 3D model', self.reset_3D_Model_eye1 ))

        self.menu.append(pupil1_menu)

        glint_menu = ui.Growing_Menu('Glint0')
        glint_menu.collapsed = True
        glint_menu.append(ui.Slider('glint_dist',self,label='Distance from pupil',min=0,max=5,step=0.25))
        glint_menu.append(ui.Slider('glint_thres',self,label='Intensity offset',min=0,max=255,step=1))
        glint_menu.append(ui.Slider('glint_min',self,label='Min size',min=1,max=250,step=1))
        glint_menu.append(ui.Slider('glint_max',self,label='Max size',min=50,max=1000,step=5))
        glint_menu.append(ui.Slider('dilate',self,label='Dilate',min=0,max=2,step=1))
        self.menu.append(glint_menu)

        glint_menu1 = ui.Growing_Menu('Glint1')
        glint_menu1.collapsed = True
        glint_menu1.append(ui.Slider('glint_dist1',self,label='Distance from pupil',min=0,max=5,step=0.25))
        glint_menu1.append(ui.Slider('glint_thres1',self,label='Intensity offset',min=0,max=255,step=1))
        glint_menu1.append(ui.Slider('glint_min1',self,label='Min size',min=1,max=250,step=1))
        glint_menu1.append(ui.Slider('glint_max1',self,label='Max size',min=50,max=1000,step=5))
        glint_menu1.append(ui.Slider('dilate1',self,label='Dilate',min=0,max=2,step=1))
        self.menu.append(glint_menu1)


        self.menu.append(ui.Button("Recalculate pupils", self.recalculate))

        self.menu.append(ui.Info_Text('Show the eye video overlaid on top of the world video. Eye1 is usually the right eye'))
        self.menu.append(ui.Slider('alpha',self,min=0.0,step=0.05,max=1.0,label='Opacity'))
        self.menu.append(ui.Slider('eye_scale_factor',self,min=0.2,step=0.1,max=1.0,label='Video Scale'))
        self.menu.append(ui.Switch('move_around',self,label="Move Overlay"))
        if len(self.eye_cap) == 2:
            self.menu.append(ui.Selector('showeyes',self,label='Show',selection=[(0,),(1,),(0,1)],labels= ['Eye 0','Eye 1','both'],setter=self.set_showeyes))
        if 0 in self.showeyes:
            self.menu.append(ui.Switch('0',self.mirror,label="Eye 0: Horiz. Flip"))
            self.menu.append(ui.Switch('0',self.flip,label="Eye 0: Vert. Flip"))
        if 1 in self.showeyes:
            self.menu.append(ui.Switch('1',self.mirror,label="Eye 1: Horiz Flip"))
            self.menu.append(ui.Switch('1',self.flip,label="Eye 1: Vert Flip"))

    def reset_3D_Model_eye0(self):
        self.pupil_detectors3D[0].reset_3D_Model()

    def reset_3D_Model_eye1(self):
        self.pupil_detectors3D[1].reset_3D_Model()


    def setPupilDetectors(self):
        if self.detect_3D == 1:
            self.pupil_detectors = self.pupil_detectors3D
        else:
            self.pupil_detectors = self.pupil_detectors2D


    def setSettings(self, eye_index, settings, glint_settings):

        if eye_index == 0:
            settings["intensity_range"] = self.intensity_range
            settings["pupil_size_min"] = self.pupil_size_min
            settings["pupil_size_max"] = self.pupil_size_max
            settings['model_sensitivity'] = self.model_sensitivity
            settings["ellipse_roundness_ratio"] = self.ellipse_roundness_ratio
            settings["coarse_filter_min"] = self.coarse_filter_min
            settings["coarse_filter_max"] = self.coarse_filter_max
            settings["canny_treshold"] = self.canny_treshold
            settings["canny_ration"] = self.canny_ration

            if self.detect_3D:
                settings_copy = dict(settings)
                del settings_copy['2D_Settings']
                del settings_copy['3D_Settings']
                settings['2D_Settings'].update(settings_copy)
                settings['3D_Settings'].update(settings_copy)

            glint_settings['glint_dist'] = self.glint_dist
            glint_settings['glint_thres'] = self.glint_thres
            glint_settings['glint_min'] = self.glint_min
            glint_settings['glint_max'] = self.glint_max
            glint_settings['dilate'] = self.dilate
            glint_settings['erode'] = self.erode

            self.current_settings_0.update(settings)

        if eye_index == 1:
            settings["intensity_range"] = self.intensity_range1
            settings["pupil_size_min"] = self.pupil_size_min1
            settings["pupil_size_max"] = self.pupil_size_max1
            settings['model_sensitivity'] = self.model_sensitivity1
            settings["ellipse_roundness_ratio"] = self.ellipse_roundness_ratio1
            settings["coarse_filter_min"] = self.coarse_filter_min1
            settings["coarse_filter_max"] = self.coarse_filter_max1
            settings["canny_treshold"] = self.canny_treshold1
            settings["canny_ration"] = self.canny_ration1


            if self.detect_3D:
                settings_copy = dict(settings)
                del settings_copy['2D_Settings']
                del settings_copy['3D_Settings']
                settings['2D_Settings'].update(settings_copy)
                settings['3D_Settings'].update(settings_copy)

            glint_settings['glint_dist'] = self.glint_dist1
            glint_settings['glint_thres'] = self.glint_thres1
            glint_settings['glint_min'] = self.glint_min1
            glint_settings['glint_max'] = self.glint_max1
            glint_settings['dilate'] = self.dilate1
            glint_settings['erode'] = self.erode1

            self.current_settings_1.update(settings)


    def calculate_pupil(self,eye_index, ts):
        self.recalculating += 1
        pupil_detector = self.pupil_detectors[eye_index]
        glint_detector = self.glint_detectors[eye_index]

        settings = pupil_detector.get_settings()
        glint_settings = glint_detector.settings()
        self.setSettings(eye_index, settings, glint_settings)
        glint_detector.update()

        self.gPool0.pupil_queue = Queue()
        self.gPool1.pupil_queue = Queue()

        using3D = self.detect_3D
        model_birth_timestamp = -1

        calibTime = float("inf")
        updating = False
        inPast = False
        present = 0
        lastCalibTime = float("inf")
        lastCalibFrame = float("inf")
        try:
            fp = os.path.join(self.rec_dir,"calibTimes.npy")
            logger.info(fp)
            calibTimes = np.load(fp).tolist()
            calibTime = calibTimes.pop(0)
            logger.info(calibTimes)
        except:
            calibTimes = [0]
            calibTime = calibTimes.pop(0)
            logger.info("No calibration times")
            pass

        timestamps = np.load(ts)
        data = {'pupil_positions':[]}
        self.eye_cap[eye_index].seek_to_frame(0)

        i = 0
        while True:
            
            if i == timestamps.size:
                break
            if i % 1000 == 0:
                logger.info("eye %d: %d frames processed" % (eye_index, i))
            t = timestamps[i]
            image = self.eye_cap[eye_index].get_frame()
            result,roi = pupil_detector.detect(image, self.u_r[eye_index], False)
            glints = [[0,0,0,0,0,0], [0,0,0,0,0,1]] #glint_detector.glint(image, eye_index, u_roi=self.u_r, pupil=result, roi=roi)
            result['glints'] = glints
            result['id'] = eye_index
            result['timestamp'] = t
            result['updating'] = updating
                
            data['pupil_positions'].append(result)

            center = result['ellipse']['center']
            center = [int(x) for x in center]
            
            if eye_index == 0:
                self.gPool0.pupil_queue.put(result)
            else:
                self.gPool1.pupil_queue.put(result)

            if using3D and model_birth_timestamp != result['model_birth_timestamp']:
                model_birth_timestamp = result['model_birth_timestamp']
                lastCalibTime = result['model_birth_timestamp']
                if not updating:
                    updating = True
                    lastCalibFrame = i
                logger.info("eye %d: 3D Model update at frame %d" % (eye_index, i))

            if inPast and t >= present:
                inPast = False

            if using3D and (t > calibTime or (result['projected_sphere']['axes'][0] < 300 and t > lastCalibTime + 20)) and not inPast:
                pupil_detector.reset_3D_Model()
                if len(calibTimes) > 0:
                    calibTime = calibTimes.pop(0)
                else:
                     calibTime = float("inf")
                logger.info("eye %d: 3D Model manual reset at frame %d" % (eye_index, i))

            if t > (lastCalibTime + 30) or (i == timestamps.size -1 and updating == True):
                i = lastCalibFrame
                self.eye_cap[eye_index].seek_to_frame(i)
                updating = False
                inPast = True
                present = t
                lastCalibTime = float("inf")
                lastCalibFrame = float("inf")
                logger.info("eye %d: returning to the past" % eye_index)
            else:
                i += 1
        settings = pupil_detector.get_settings()
        settings['roi'] = self.u_r[eye_index].get()
        data['pupil_detector_settings'] = settings
        save_object(data,os.path.join(self.rec_dir,"recalculated_pupil_" + str(eye_index)))
        logger.debug("eye %d finished" % eye_index)
        self.recalculating -= 1
        #self.threads[eye_index].join()

    def recalculate(self):
        if self.recalculating > 0:
             logger.warning("Already recalculating")
        else:
            self.setPupilDetectors()
            self.threads = [[],[]]
            for eye_index, ts in zip(self.showeyes, self.eye_timestamps_path):
                self.threads[eye_index] = _thread.start_new_thread(self.calculate_pupil, (eye_index, ts) )

    def set_showeyes(self,new_mode):
        #everytime we choose eye setting (either use eye 1, 2, or both, updates the gui menu to remove certain options from list)
        self.showeyes = new_mode
        self.update_gui()

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def update(self, frame, events):
        for eye_index in self.showeyes:
            requested_eye_frame_idx = self.eye_world_frame_map[eye_index][frame.index]

            #1. do we need a new frame?
            if (requested_eye_frame_idx != self.eye_frames[eye_index].index or self.urActive) and self.recalculating == 0:
                # do we need to seek?
                if requested_eye_frame_idx == self.eye_cap[eye_index].get_frame_index()+1:
                    # if we just need to seek by one frame, its faster to just read one and and throw it away.
                    _ = self.eye_cap[eye_index].get_frame()
                if requested_eye_frame_idx != self.eye_cap[eye_index].get_frame_index():
                    # only now do I need to seek
                    self.eye_cap[eye_index].seek_to_frame(requested_eye_frame_idx)
                # reading the new eye frame frame
                try:
                    self.eye_frames[eye_index] = self.eye_cap[eye_index].get_frame()
                except EndofVideoFileError:
                    logger.warning("Reached the end of the eye video for eye video {}.".format(eye_index))
            else:
                #our old frame is still valid because we are doing upsampling
                pass



            #2. dragging image
            if self.drag_offset[eye_index] is not None:
                pos = glfwGetCursorPos(glfwGetCurrentContext())
                pos = normalize(pos,glfwGetWindowSize(glfwGetCurrentContext()))
                pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
                self.pos[eye_index][0] = pos[0]+self.drag_offset[eye_index][0]
                self.pos[eye_index][1] = pos[1]+self.drag_offset[eye_index][1]
            else:
                self.video_size = [round(self.eye_frames[eye_index].width*self.eye_scale_factor), round(self.eye_frames[eye_index].height*self.eye_scale_factor)]

            pos = glfwGetCursorPos(glfwGetCurrentContext())
            pos = normalize(pos,glfwGetWindowSize(glfwGetCurrentContext()))
            pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) )
            pos = (int(pos[0] - self.pos[eye_index][0]), int(pos[1] - self.pos[eye_index][1]))
            self.mouse_img[eye_index] = pos
            
            if self.urActive == eye_index + 1:
                self.u_r[eye_index].move_vertex(self.u_r[eye_index].active_pt_idx,pos)

            if self.recalculating == 0:
                self.setPupilDetectors()
                pupil_detector = self.pupil_detectors[eye_index]
                glint_detector = self.glint_detectors[eye_index]
        
                settings = pupil_detector.get_settings()

                glint_settings = glint_detector.settings()
                self.setSettings(eye_index, settings, glint_settings)
                glint_detector.update()


                new_frame = self.eye_frames[eye_index]
                if self.algorithm == 1:
                  view = "algorithm"
                else:
                  view = False
                result, roi = pupil_detector.detect(new_frame, self.u_r[eye_index], view)
                glints = [[0,0,0,0,0,0], [0,0,0,0,0,1]] #glint_detector.glint(new_frame, eye_index, u_roi=self.u_r, pupil=result, roi=roi)

                if eye_index == 0:
                    self.gPool0.pupil_queue.put(result)
                else:
                    self.gPool1.pupil_queue.put(result)

                #3. keep in image bounds, do this even when not dragging because the image video_sizes could change.
                #self.pos[eye_index][1] = min(frame.img.shape[0]-self.video_size[1],max(self.pos[eye_index][1],0)) #frame.img.shape[0] is height, frame.img.shape[1] is width of screen
                #self.pos[eye_index][0] = min(frame.img.shape[1]-self.video_size[0],max(self.pos[eye_index][0],0))

                #4. flipping images, converting to greyscale
                #eye_gray = cv2.cvtColor(self.eye_frames[eye_index].img,cv2.COLOR_BGR2GRAY) #auto gray scaling
                pts = cv2.ellipse2Poly( (int(result['ellipse']['center'][0]),int(result['ellipse']['center'][1])),
                                                (int(result['ellipse']['axes'][0]/2),int(result['ellipse']['axes'][1]/2)),
                                                int(result['ellipse']['angle']),0,360,15)
                cv2.polylines(self.eye_frames[eye_index].img, [pts], 1, (0,255,0))
                center = result['ellipse']['center']
                center = [int(x) for x in center]
                cv2.circle(self.eye_frames[eye_index].img, tuple(center), True, (0,255,0), thickness=15)

                #img = 255 - np.mean(new_frame.img, axis=2)
                #img = (img*255).astype(np.uint8)
                #ret = pyelse.run(img, 10)
                #x = ret.center.x
                #y = ret.center.y
                #cv2.circle(self.eye_frames[eye_index].img, (int(x), int(y)), True, (255,0,0), thickness=10)


                glints = np.array(glints)
                #if len(glints)>0 and glints[0][3]:
                #    for g in glints:
                #        cv2.circle(self.eye_frames[eye_index].img, (int(g[1]),int(g[2])), True,(255,0,0),thickness=5)

                if result['method'] == '3d c++':

                        eye_ball = result['projected_sphere']
                        try:
                            pts = cv2.ellipse2Poly( (int(eye_ball['center'][0]),int(eye_ball['center'][1])),
                                                (int(eye_ball['axes'][0]/2),int(eye_ball['axes'][1]/2)),
                                                int(eye_ball['angle']),0,360,8)
                        except ValueError as e:
                            pass
                        else:
                            cv2.polylines(self.eye_frames[eye_index].img, [pts], 1, (255,0,0))

            if self.show_ellipses and events['pupil_positions'] and self.recalculating == 0:
                for pd in events['pupil_positions']:
                    if pd['id'] == eye_index and pd['timestamp'] == self.eye_frames[eye_index].timestamp:
                        break

                if pd['method'] == '3d c++':
                    eye_ball = pd['projected_sphere']
                    try:
                        pts = cv2.ellipse2Poly( (int(eye_ball['center'][0]),int(eye_ball['center'][1])),
                                            (int(eye_ball['axes'][0]/2),int(eye_ball['axes'][1]/2)),
                                            int(eye_ball['angle']),0,360,8)
                    except ValueError as e:
                        pass
                    else:
                        cv2.polylines(self.eye_frames[eye_index].img, [pts], 1, (0,255,0))

                el = pd['ellipse']
                conf = int(pd.get('model_confidence', pd.get('confidence', 0.1)) * 255)
                center = list(map(lambda val: int(val), el['center']))
                el['axes'] = tuple(map(lambda val: int(val/2), el['axes']))
                el['angle'] = int(el['angle'])
                el_points = cv2.ellipse2Poly(tuple(center), el['axes'], el['angle'], 0, 360, 1)

                cv2.polylines(self.eye_frames[eye_index].img, [np.asarray(el_points)], True, (0, 0, 255, conf), thickness=math.ceil(2))
                cv2.circle(self.eye_frames[eye_index].img, tuple(center), int(5*self.eye_scale_factor), (0, 0, 255, conf), thickness=-1)


            rect = np.asarray(self.u_r[eye_index].rect)
            #rect[:,[0, 1]] = rect[:,[1, 0]]
            cv2.polylines(self.eye_frames[eye_index].img, [rect],True, (0, 255, 255), thickness=math.ceil(2))
            for corner in rect:
                cv2.circle(self.eye_frames[eye_index].img, (int(corner[0]), int(corner[1])), int(5*self.eye_scale_factor), (0, 255, 255), thickness=-1)


            
            #3. keep in image bounds, do this even when not dragging because the image video_sizes could change.
            self.pos[eye_index][1] = min(frame.img.shape[0]-self.video_size[1],max(self.pos[eye_index][1],0)) #frame.img.shape[0] is height, frame.img.shape[1] is width of screen
            self.pos[eye_index][0] = min(frame.img.shape[1]-self.video_size[0],max(self.pos[eye_index][0],0))

            #4. flipping images, converting to greyscale
            #eye_gray = cv2.cvtColor(self.eye_frames[eye_index].img,cv2.COLOR_BGR2GRAY) #auto gray scaling
            eyeimage = cv2.resize(self.eye_frames[eye_index].img,(0,0),fx=self.eye_scale_factor, fy=self.eye_scale_factor)
            if self.mirror[str(eye_index)]:
                eyeimage = np.fliplr(eyeimage)
            if self.flip[str(eye_index)]:
                eyeimage = np.flipud(eyeimage)


            #eyeimage = cv2.cvtColor(eyeimage, cv2.COLOR_GRAY2BGR)


            # 5. finally overlay the image
            x, y = int(self.pos[eye_index][0]), int(self.pos[eye_index][1])
            transparent_image_overlay((x, y), eyeimage, frame.img, self.alpha)

    def on_click(self,pos,button,action):
        if action == 1:
            if self.urActive == False:
                for eye_index in self.showeyes:
                    if self.pos[eye_index][0] < pos[0] < self.pos[eye_index][0]+self.video_size[0] and self.pos[eye_index][1] < pos[1] < self.pos[eye_index][1] + self.video_size[1]:
                        img_pos = self.mouse_img[eye_index]
                        if self.u_r[eye_index].mouse_over_edit_pt(img_pos, self.u_r[eye_index].handle_size + 20,self.u_r[eye_index].handle_size + 20):
                            self.urActive = eye_index + 1
            else:
                self.urActive = False
        if self.move_around == 1 and action == 1:
            for eye_index in self.showeyes:
                if self.pos[eye_index][0] < pos[0] < self.pos[eye_index][0]+self.video_size[0] and self.pos[eye_index][1] < pos[1] < self.pos[eye_index][1] + self.video_size[1]:
                    self.drag_offset[eye_index] = self.pos[eye_index][0]-pos[0],self.pos[eye_index][1]-pos[1]
                    return
        else:
            self.drag_offset = [None,None]

    def get_init_dict(self):
        return {'alpha':self.alpha,'eye_scale_factor':self.eye_scale_factor,'move_around':self.move_around,'mirror':self.mirror,'flip':self.flip,'pos':self.pos,'move_around':self.move_around, 'show_ellipses': self.show_ellipses}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.persistent_settings_0.update(self.current_settings_1)
        self.persistent_settings_1.update(self.current_settings_0)
        self.persistent_settings_roi['roi0'] = self.u_r[0].get()
        self.persistent_settings_roi['roi1'] = self.u_r[1].get()
        self.persistent_settings_0.close()
        self.persistent_settings_1.close()
        self.persistent_settings_roi.close()
        self.deinit_gui()
