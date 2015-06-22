import os
import cv2
import numpy as np
import scipy.spatial as sp



from methods import normalize,denormalize
from gl_utils import draw_gl_point,adjust_gl_view,clear_gl_screen,basic_gl_setup
import OpenGL.GL as gl
from glfw import *
import calibrate
from circle_detector import get_candidate_ellipses

import audio

from pyglui import ui
from pyglui.cygl.utils import draw_points, draw_points_norm, draw_polyline, draw_polyline_norm, RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from plugin import Calibration_Plugin
from screen_marker_calibration import draw_marker,on_resize, easeInOutQuad, interp_fn, Screen_Marker_Calibration
from calibrate import preprocess_data
#logging
import logging
logger = logging.getLogger(__name__)



class Verification(Screen_Marker_Calibration,Calibration_Plugin):
     def __init__(self, g_pool,menu_conf = {'collapsed':False},fullscreen=True,marker_scale=1.0,sample_duration=40):
        super(Verification, self).__init__(g_pool,menu_conf,fullscreen,marker_scale)