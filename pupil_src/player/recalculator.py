import sys
import os
import platform

if getattr(sys, 'frozen', False):
    user_dir = os.path.expanduser(os.path.join('~', 'pupil_player_settings'))
    version_file = os.path.join(sys._MEIPASS, '_version_string_')
else:
    # We are running in a normal Python environment.
    # Make all pupil shared_modules available to this Python session.
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'capture'))

    # Specifiy user dirs.
    user_dir = os.path.join(pupil_base_dir, 'player_settings')
    version_file = None

import logging
# set up root logger before other imports
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# since we are not using OS.fork on MacOS we need to do a few extra things to log our exports correctly.
if platform.system() == 'Darwin':
    if __name__ == '__main__':  # clear log if main
        fh = logging.FileHandler(os.path.join(user_dir, 'recalc.log'), mode='w')
    # we will use append mode since the exporter will stream into the same file when using os.span processes
    fh = logging.FileHandler(os.path.join(user_dir, 'recalc.log'), mode='a')
else:
    fh = logging.FileHandler(os.path.join(user_dir, 'recalc.log'), mode='w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('Player: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
formatter = logging.Formatter('Player [%(levelname)s] %(name)s : %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

logging.getLogger("OpenGL").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

import numpy as np
from time import sleep
from vis_eye_video_overlay import Vis_Eye_Video_Overlay
import threading

class Global_Container(object):
    pass
                
if __name__ == '__main__':
    
    g_pool = Global_Container()
    rootdir = sys.argv[1]
    
    for path in os.walk(rootdir):
        path = str(path[0]) + "/"
        if os.path.exists(path + "/pupil_data"):

            g_pool.user_dir = path
            g_pool.rec_dir = path
            g_pool.meta_info = {}
            g_pool.meta_info['Capture Software Version'] = '10.8' #needs to be higher than 0.4. otherwise doesn't matter
            g_pool.timestamps = np.load(path + "/world_timestamps.npy")  #world timestamps are not actually used for anything, but vis_eye_video_overylay requires them
            recal = Vis_Eye_Video_Overlay(g_pool)
            recal.detect_3D = 0

            #create and start threads for 2D detection
            eye0 = threading.Thread(target=recal.calculate_pupil, args=(0, recal.eye_timestamps_path[0]))
            eye1 = threading.Thread(target=recal.calculate_pupil, args=(1, recal.eye_timestamps_path[1]))
            eye0.start()
            eye1.start()

            #make sure the threads actually start before switching to 3D
            sleep(0.3)
            
            #create and start threads for 3D detection
            recal.detect_3D = 1
            recal.setPupilDetectors()

            #if you want to set default values for model sensitivity
            recal.model_sensitivity, recal.model_sensitivity1 = 0.997, 0.997
            
            eye03D = threading.Thread(target=recal.calculate_pupil, args=(0, recal.eye_timestamps_path[0]))
            eye13D = threading.Thread(target=recal.calculate_pupil, args=(1, recal.eye_timestamps_path[1]))
            eye03D.start()
            eye13D.start()

            eye0.join()
            eye1.join()
            eye03D.join()
            eye13D.join()
    print("all videos should have been processed")
