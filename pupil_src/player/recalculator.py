import main
import numpy as np
from time import sleep
from vis_eye_video_overlay import Vis_Eye_Video_Overlay
import threading
import os, sys

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
