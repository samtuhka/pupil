'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
import cv2
from . import calibrate
from methods import project_distort_pts , normalize, spherical_to_cart
from copy import deepcopy
import numpy as np
from pyglui import ui
import math_helper
import zmq_tools
import threading
from multiprocessing import Process as Thread

from . visualizer_calibration import Calibration_Visualizer

def _clamp_norm_point(pos):
    '''realistic numbers for norm pos should be in this range.
        Grossly bigger or smaller numbers are results bad exrapolation
        and can cause overflow erorr when denormalized and cast as int32.
    '''
    return min(100.,max(-100.,pos[0])),min(100.,max(-100.,pos[1]))

class Gaze_Mapping_Plugin(Plugin):
    '''base class for all gaze mapping routines'''
    uniqueness = 'by_base_class'
    order = .1
    def __init__(self,g_pool):
        super().__init__(g_pool)
        self.g_pool.active_gaze_mapping_plugin = self

    def on_pupil_datum(self, p):
        raise NotImplementedError()


class Monocular_Gaze_Mapper_Base(Gaze_Mapping_Plugin):
    """Base class to implement the map callback"""
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.min_pupil_confidence = 0.0

    def on_pupil_datum(self, p):
        if p['confidence'] >= self.min_pupil_confidence:
            g = self._map_monocular(p)
            if g:
                return [g,]
            else:
                return []
        else:
            return []

class Binocular_Gaze_Mapper_Base(Gaze_Mapping_Plugin):
    """Base class to implement the map callback"""
    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.min_pupil_confidence = 0.6
        self._caches = ([],[])
        self.temportal_cutoff = 0.3
        self.sample_cutoff = 10

    def on_pupil_datum(self, p):
        if p['confidence'] >= self.min_pupil_confidence:
            self._caches[p['id']].append(p)

        if self._caches[0] and self._caches[1]:
            #we have binocular data

            if self._caches[0][0]['timestamp'] < self._caches[1][0]['timestamp']:
                p0 = self._caches[0].pop(0)
                p1 = self._caches[1][0]
                older_pt = p0
            else:
                p0 = self._caches[0][0]
                p1 = self._caches[1].pop(0)
                older_pt = p1

            if abs(p0['timestamp'] - p1['timestamp']) < self.temportal_cutoff:
                gaze_datum = self._map_binocular(p0,p1)
            else:
                gaze_datum = self._map_monocular(older_pt)

        elif len(self._caches[0])>self.sample_cutoff:
            p = self._caches[0].pop(0)
            gaze_datum = self._map_monocular(p)
        elif len(self._caches[1])>self.sample_cutoff:
            p = self._caches[1].pop(0)
            gaze_datum = self._map_monocular(p)
        else:
            gaze_datum = None

        if gaze_datum:
            return [gaze_datum,]
        else:
            return []



class Dummy_Gaze_Mapper(Monocular_Gaze_Mapper_Base,Gaze_Mapping_Plugin):
    """docstring for Dummy_Gaze_Mapper"""
    def __init__(self, g_pool):
        super().__init__(g_pool)

    def _map_monocular(self,p):
        return {'topic':'gaze','norm_pos':p['norm_pos'],'confidence':p['confidence'],'timestamp':p['timestamp'],'base_data':[p]}

    def get_init_dict(self):
        return {}


class Monocular_Gaze_Mapper(Monocular_Gaze_Mapper_Base,Gaze_Mapping_Plugin):
    """docstring for Monocular_Gaze_Mapper"""
    def __init__(self, g_pool,params):
        super().__init__(g_pool)
        self.params = params
        self.map_fn = calibrate.make_map_function(*self.params)

    def _map_monocular(self,p):
        gaze_point = self.map_fn(p['norm_pos'])
        return {'topic':'gaze','norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base_data':[p]}


    def get_init_dict(self):
        return {'params':self.params}


class Dual_Monocular_Gaze_Mapper(Monocular_Gaze_Mapper_Base,Gaze_Mapping_Plugin):
    """A gaze mapper that maps two eyes individually"""
    def __init__(self, g_pool,params0,params1):
        super().__init__(g_pool)
        self.params0 = params0
        self.params1 = params1
        self.map_fns = (calibrate.make_map_function(*self.params0),calibrate.make_map_function(*self.params1))

    def _map_monocular(self,p):
        gaze_point = self.map_fns[p['id']](p['norm_pos'])
        return {'topic':'gaze','norm_pos':gaze_point,'confidence':p['confidence'],'id':p['id'],'timestamp':p['timestamp'],'base_data':[p]}

    def get_init_dict(self):
        return {'params0':self.params0,'params1':self.params1}


class Binocular_Gaze_Mapper(Binocular_Gaze_Mapper_Base,Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params,params_eye0,params_eye1):
        super().__init__(g_pool)
        self.params = params
        self.params_eye0 = params_eye0
        self.params_eye1 = params_eye1
        self.multivariate = True
        self.map_fn = calibrate.make_map_function(*self.params)
        self.map_fn_fallback = []
        self.map_fn_fallback.append(calibrate.make_map_function(*self.params_eye0))
        self.map_fn_fallback.append(calibrate.make_map_function(*self.params_eye1))


    def init_gui(self):
        self.menu = ui.Growing_Menu('Binocular Gaze Mapping')
        self.g_pool.sidebar.insert(3,self.menu)
        self.menu.append(ui.Switch('multivariate',self,label='Multivariate Mode'))


    def _map_binocular(self, p0, p1):
        if self.multivariate:
            gaze_point = self.map_fn(p0['norm_pos'], p1['norm_pos'])
        else:
            gaze_point_eye0 = self.map_fn_fallback[0](p0['norm_pos'])
            gaze_point_eye1 = self.map_fn_fallback[1](p1['norm_pos'])
            gaze_point = (gaze_point_eye0[0] + gaze_point_eye1[0])/2. , (gaze_point_eye0[1] + gaze_point_eye1[1])/2.
        confidence = (p0['confidence'] + p1['confidence'])/2.
        ts = (p0['timestamp'] + p1['timestamp'])/2.
        return {'topic':'gaze','norm_pos':gaze_point,'confidence':confidence,'timestamp':ts,'base_data':[p0, p1]}

    def _map_monocular(self,p):
        gaze_point = self.map_fn_fallback[p['id']](p['norm_pos'])
        return {'topic':'gaze','norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base_data':[p]}

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def cleanup(self):
        super().cleanup()
        self.deinit_gui()

    def get_init_dict(self):
        return {'params':self.params, 'params_eye0':self.params_eye0, 'params_eye1':self.params_eye1}

class Glint_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Simple_Gaze_Mapper"""
    def __init__(self, g_pool, params, interpolParams):
        super(Glint_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fn = calibrate.make_map_function_two_glints(*params)
        self.interpol_params = interpolParams
        self.interpol_map = calibrate.make_map_function(*interpolParams)

    def update(self,frame,events):
        """
        gaze_pts = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_point = self.map_fn(p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp']})

        events['gaze'] = gaze_pts
        """

        gaze_pts = []
        for g in events['glint_pupil_vectors']:
            if g['pupil_confidence'] > self.g_pool.pupil_confidence_threshold:
                v = g['x'], g['y'], g['x2'], g['y2']
                if g['glint_found']:
                    gaze_glint_point = self.map_fn(v)
                else:
                    gaze_glint_point = self.interpol_map((g['x'], g['y']))
                gaze_pts.append({'norm_pos':gaze_glint_point,'confidence':g['pupil_confidence'],'timestamp':g['timestamp'], 'foundGlint': g['glint_found'], 'id': g['id']})
        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params, 'interpolParams': self.interpol_params}

class Binocular_Gaze_Mapper_Old(Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params):
        super(Binocular_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fns = (calibrate.make_map_function(*self.params[0:3]),calibrate.make_map_function(*self.params[3:6]))

    def update(self,frame,events):
        gaze_pts = []
        gaze_mono_pts = [[],[]]

        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                eye_id = p['id']
                gaze_point = self.map_fns[eye_id](p['norm_pos'])
                gaze_mono_pts[eye_id].append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'], 'id': eye_id})

        # Pair gaze positions and compute means
        i = 0
        j = 0
        while  i < len(gaze_mono_pts[0]) and len(gaze_mono_pts[1]) == 0:
            gaze_pts.append(gaze_mono_pts[0][i])
            i += 1
        while  j < len(gaze_mono_pts[1]) and len(gaze_mono_pts[0]) == 0:
            gaze_pts.append(gaze_mono_pts[1][j])
            j += 1
        while i < len(gaze_mono_pts[0]) and j < len(gaze_mono_pts[1]):
            gaze_0 = gaze_mono_pts[0][i]
            gaze_1 = gaze_mono_pts[1][j]
            diff = gaze_0['timestamp'] - gaze_1['timestamp']
            if abs(diff) <= 1/15.: #assuming 30fps + slack
                x_0, y_0 = gaze_0['norm_pos']
                x_1, y_1 = gaze_1['norm_pos']
                gaze_point = ((x_0+x_1)/2,(y_0+y_1)/2)
                confidence = min(gaze_0['confidence'], gaze_1['confidence'])
                timestamp = max(gaze_0['timestamp'], gaze_1['timestamp'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':confidence,'timestamp':timestamp, 'id': 2})
                i += 1
                j += 1
            elif diff > 0:
                j += 1
            else:
                i += 1
        events['gaze_positions'] = gaze_pts
        events['gaze_eye_0'] = gaze_mono_pts[0]
        events['gaze_eye_1'] = gaze_mono_pts[1]

    def get_init_dict(self):
        return {'params':self.params}

class Binocular_Glint_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params, interpolParams):
        super(Binocular_Glint_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.paramsInterpol = interpolParams
        self.map_fns = (calibrate.make_map_function_two_glints(*self.params[0:3]),calibrate.make_map_function_two_glints(*self.params[3:6]))
        self.map_fns_interpol = (calibrate.make_map_function(*self.paramsInterpol[0:3]),calibrate.make_map_function(*self.paramsInterpol[3:6]))

    def update(self,frame,events):
        gaze_pts = []
        gaze_mono_pts = [[],[]]
        for g in events['glint_pupil_vectors']:
            if g['pupil_confidence'] > self.g_pool.pupil_confidence_threshold:
                if g['glint_found']:
                    eye_id = g['id']
                    v = g['x'], g['y'], g['x2'], g['y2']
                    gaze_glint_point = self.map_fns[eye_id](v)
                    gaze_mono_pts[eye_id].append({'norm_pos':gaze_glint_point,'confidence':g['pupil_confidence'],'timestamp':g['timestamp'], 'foundGlint': g['glint_found'], 'id': g['id']})

        i = 0
        j = 0
        while  i < len(gaze_mono_pts[0]) and len(gaze_mono_pts[1]) == 0:
            gaze_pts.append(gaze_mono_pts[0][i])
            i += 1
        while  j < len(gaze_mono_pts[1]) and len(gaze_mono_pts[0]) == 0:
            gaze_pts.append(gaze_mono_pts[1][j])
            j += 1

        while i < len(gaze_mono_pts[0]) and j < len(gaze_mono_pts[1]):
            gaze_0 = gaze_mono_pts[0][i]
            gaze_1 = gaze_mono_pts[1][j]
            diff = gaze_0['timestamp'] - gaze_1['timestamp']
            if abs(diff) <= 1/15.: #assuming 30fps + slack
                x_0, y_0 = gaze_0['norm_pos']
                x_1, y_1 = gaze_1['norm_pos']
                gaze_point = ((x_0+x_1)/2,(y_0+y_1)/2)
                confidence = min(gaze_0['confidence'], gaze_1['confidence'])
                timestamp = (gaze_0['timestamp'] + gaze_1['timestamp'])/2
                gaze_pts.append({'norm_pos':gaze_point,'confidence':confidence,'timestamp':timestamp, 'foundGlint': True, 'id': 2})
                i += 1
                j += 1
            elif diff > 0:
                j += 1
            else:
                i += 1

        events['gaze_positions'] = gaze_pts
        events['gaze_eye_0'] = gaze_mono_pts[0]
        events['gaze_eye_1'] = gaze_mono_pts[1]

    def get_init_dict(self):
        return {'params':self.params, 'interpolParams': self.paramsInterpol}


class Bilateral_Glint_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params,params_eye0,params_eye1):
        super(Gaze_Mapping_Plugin, self).__init__(g_pool)
        self.params = params
        self.params_eye0 = params_eye0
        self.params_eye1 = params_eye1
        self.multivariate = False
        self.map_fn = calibrate.make_map_function_two_glints(*self.params)
        self.map_fn_fallback = []
        self.map_fn_fallback.append(calibrate.make_map_function_two_glints(*self.params_eye0))
        self.map_fn_fallback.append(calibrate.make_map_function_two_glints(*self.params_eye1))

    def init_gui(self):
        self.menu = ui.Growing_Menu('Binocular Gaze Mapping')
        self.g_pool.sidebar.insert(3,self.menu)
        self.menu.append(ui.Switch('multivariate',self,on_val=True,off_val=False,label='Multivariate Mode'))

    def update(self,frame,events):

        glint_pts_0 = []
        glint_pts_1 = []
        for g in events['glint_pupil_vectors']:
            if g['pupil_confidence'] > self.g_pool.pupil_confidence_threshold and g['glint_found']:
                if g['id'] == 0:
                    glint_pts_0.append(g)
                else:
                    glint_pts_1.append(g)

        # try binocular mapping (needs at least 1 pupil position in each list)
        gaze_pts = []
        if len(glint_pts_0) > 0 and len(glint_pts_1) > 0:
            gaze_pts = self._map_binocular(glint_pts_0, glint_pts_1, self.multivariate)
        # fallback to monocular if something went wrong
        else:
            for g in glint_pts_0:
                v = g['x'], g['y'], g['x2'], g['y2']
                gaze_glint_point = self.map_fn_fallback[0](v)
                gaze_pts.append({'norm_pos':gaze_glint_point,'confidence':g['pupil_confidence'],'timestamp':g['timestamp'], 'foundGlint': g['glint_found'], 'id': g['id']})
            for g in glint_pts_1:
                v = g['x'], g['y'], g['x2'], g['y2']
                gaze_glint_point = self.map_fn_fallback[1](v)
                gaze_pts.append({'norm_pos':gaze_glint_point,'confidence':g['pupil_confidence'],'timestamp':g['timestamp'], 'foundGlint': g['glint_found'], 'id': g['id']})

        events['gaze_positions'] = gaze_pts

    def _map_binocular(self, glint_pts_0, glint_pts_1,multivariate=True):
        # maps gaze with binocular mapping
        # requires each list to contain at least one item!
        # returns 1 gaze point at minimum
        gaze_pts = []
        g0 = glint_pts_0.pop(0)
        g1 = glint_pts_1.pop(0)
        while True:
            v0 = g0['x'], g0['y'], g0['x2'], g0['y2']
            v1 = g1['x'], g1['y'], g1['x2'], g1['y2']
            if multivariate:
                gaze_point = self.map_fn(v0, v1)
            else:
                gaze_point_eye0 = self.map_fn_fallback[0](v0)
                gaze_point_eye1 = self.map_fn_fallback[1](v1)
                gaze_point = (gaze_point_eye0[0] + gaze_point_eye1[0])/2. , (gaze_point_eye0[1] + gaze_point_eye1[1])/2.
            confidence = (g0['pupil_confidence'] + g1['pupil_confidence'])/2.
            ts = (g0['timestamp'] + g1['timestamp'])/2.
            gaze_pts.append({'norm_pos':gaze_point,'confidence':confidence,'timestamp':ts,'foundGlint': True, 'id': 2})

            # keep sample with higher timestamp and increase the one with lower timestamp
            if g0['timestamp'] <= g1['timestamp'] and glint_pts_0:
                p0 = glint_pts_0.pop(0)
                continue
            elif g1['timestamp'] <= g0['timestamp'] and glint_pts_1:
                g1 = glint_pts_1.pop(0)
                continue
            elif glint_pts_0 and not glint_pts_1:
                g0 = glint_pts_0.pop(0)
            elif glint_pts_1 and not glint_pts_0:
                g1 = glint_pts_1.pop(0)
            else:
                break

        return gaze_pts

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def cleanup(self):
        self.deinit_gui()

    def get_init_dict(self):
        return {'params':self.params, 'params_eye0':self.params_eye0, 'params_eye1':self.params_eye1}



class Vector_Gaze_Mapper(Monocular_Gaze_Mapper_Base,Gaze_Mapping_Plugin):
    """docstring for Vector_Gaze_Mapper"""
    def __init__(self, g_pool, eye_camera_to_world_matrix , camera_intrinsics ,cal_points_3d, cal_ref_points_3d, cal_gaze_points_3d, gaze_distance = 500 ):
        super().__init__(g_pool)
        self.eye_camera_to_world_matrix = np.asarray(eye_camera_to_world_matrix)
        self.rotation_matrix = self.eye_camera_to_world_matrix[:3,:3]
        self.rotation_vector = cv2.Rodrigues(self.rotation_matrix  )[0]
        self.translation_vector = self.eye_camera_to_world_matrix[:3,3]
        self.camera_matrix = np.asarray(camera_intrinsics['camera_matrix'])
        self.dist_coefs = np.asarray(camera_intrinsics['dist_coefs'])
        self.world_frame_size = camera_intrinsics['resolution']
        self.camera_intrinsics = camera_intrinsics
        self.cal_points_3d = cal_points_3d
        self.cal_ref_points_3d = cal_ref_points_3d
        self.cal_gaze_points_3d = cal_gaze_points_3d
        self.visualizer = Calibration_Visualizer(g_pool, camera_intrinsics ,cal_points_3d, cal_ref_points_3d,eye_camera_to_world_matrix, cal_gaze_points_3d)
        self.g_pool = g_pool
        self.gaze_pts_debug = []
        self.sphere = {}
        self.gaze_distance = gaze_distance


    def toWorld(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix , point)[:3]

    def init_gui(self):

        def open_close_window(new_state):
            if new_state:
                self.visualizer.open_window()
            else:
                self.visualizer.close_window()

        self.menu = ui.Growing_Menu('Monocular 3D gaze mapper')
        self.g_pool.sidebar.insert(3,self.menu)
        self.menu.append(ui.Switch('debug window',setter=open_close_window, getter=lambda: bool(self.visualizer.window) ))
        self.menu.append(ui.Slider('gaze_distance',self,min=50,max=2000,label='gaze distance mm'))


    def _map_monocular(self,p):
        if '3d' not in p['method']:
            return None

        gaze_point =  np.array(p['circle_3d']['normal'] ) * self.gaze_distance  + np.array( p['sphere']['center'] )

        image_point, _  =  cv2.projectPoints( np.array([gaze_point]) , self.rotation_vector, self.translation_vector , self.camera_matrix , self.dist_coefs )
        image_point = image_point.reshape(-1,2)
        image_point = normalize( image_point[0], self.world_frame_size , flip_y = True)
        image_point = _clamp_norm_point(image_point)

        eye_center = self.toWorld(p['sphere']['center'])
        gaze_3d = self.toWorld(gaze_point)
        normal_3d = np.dot( self.rotation_matrix, np.array( p['circle_3d']['normal'] ) )

        g = {   'topic':'gaze',
                'norm_pos':image_point,
                'eye_center_3d':eye_center.tolist(),
                'gaze_normal_3d':normal_3d.tolist(),
                'gaze_point_3d':gaze_3d.tolist(),
                'confidence':p['confidence'],
                'timestamp':p['timestamp'],
                'base_data':[p]}

        if self.visualizer.window:
            self.gaze_pts_debug.append( gaze_3d )
            self.sphere['center'] = eye_center #eye camera coordinates
            self.sphere['radius'] = p['sphere']['radius']
        return g


    def gl_display(self):
        self.visualizer.update_window( self.g_pool , self.gaze_pts_debug , self.sphere)
        self.gaze_pts_debug = []

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
       return {'eye_camera_to_world_matrix':self.eye_camera_to_world_matrix.tolist() ,'cal_points_3d':self.cal_points_3d,'cal_ref_points_3d':self.cal_ref_points_3d, 'cal_gaze_points_3d':self.cal_gaze_points_3d,  "camera_intrinsics":self.camera_intrinsics, 'gaze_distance':self.gaze_distance}

    def cleanup(self):
        super().cleanup()
        self.deinit_gui()
        self.visualizer.close_window()


class Binocular_Vector_Gaze_Mapper(Binocular_Gaze_Mapper_Base,Gaze_Mapping_Plugin):
    """docstring for Vector_Gaze_Mapper"""
    def __init__(self, g_pool, eye_camera_to_world_matrix0, eye_camera_to_world_matrix1 , camera_intrinsics , cal_points_3d = [],cal_ref_points_3d = [], cal_gaze_points0_3d = [], cal_gaze_points1_3d = [] ):
        super().__init__(g_pool)

        self.eye_camera_to_world_matricies = np.asarray(eye_camera_to_world_matrix0), np.asarray(eye_camera_to_world_matrix1)
        self.rotation_matricies = self.eye_camera_to_world_matricies[0][:3,:3],self.eye_camera_to_world_matricies[1][:3, :3]
        self.rotation_vectors = cv2.Rodrigues(self.eye_camera_to_world_matricies[0][:3,:3]  )[0] , cv2.Rodrigues( self.eye_camera_to_world_matricies[1][:3,:3])[0]
        self.translation_vectors  = self.eye_camera_to_world_matricies[0][:3, 3], self.eye_camera_to_world_matricies[1][:3, 3]


        self.cal_points_3d = cal_points_3d
        self.cal_ref_points_3d = cal_ref_points_3d

        self.cal_gaze_points0_3d = cal_gaze_points0_3d #save for debug window
        self.cal_gaze_points1_3d = cal_gaze_points1_3d #save for debug window

        self.camera_matrix = np.asarray(camera_intrinsics['camera_matrix'])
        self.dist_coefs = np.asarray(camera_intrinsics['dist_coefs'])
        self.world_frame_size = camera_intrinsics['resolution']
        self.camera_intrinsics = camera_intrinsics
        self.visualizer = Calibration_Visualizer(g_pool,
                                                world_camera_intrinsics=camera_intrinsics ,
                                                cal_ref_points_3d = cal_points_3d,
                                                cal_observed_points_3d = cal_ref_points_3d,
                                                eye_camera_to_world_matrix0=  self.eye_camera_to_world_matricies[0],
                                                cal_gaze_points0_3d = cal_gaze_points0_3d,
                                                eye_camera_to_world_matrix1=  self.eye_camera_to_world_matricies[1],
                                                cal_gaze_points1_3d =  cal_gaze_points1_3d)
        self.g_pool = g_pool
        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []
        self.sphere0 = {}
        self.sphere1 = {}
        self.last_gaze_distance = 500.


    def eye0_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matricies[0] , point)[:3]

    def eye1_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matricies[1] , point)[:3]


    def init_gui(self):

        def open_close_window(new_state):
            if new_state:
                self.visualizer.open_window()
            else:
                self.visualizer.close_window()

        self.menu = ui.Growing_Menu('Binocular 3D gaze mapper')
        self.g_pool.sidebar.insert(3,self.menu)
        # self.menu.append(ui.Text_Input('last_gaze_distance',self))
        self.menu.append(ui.Switch('debug window',setter=open_close_window, getter=lambda: bool(self.visualizer.window) ))

    def _map_monocular(self,p):
        if '3d' not in p['method']:
            return None

        p_id = p['id']
        gaze_point =  np.array(p['circle_3d']['normal'] ) * self.last_gaze_distance  + np.array( p['sphere']['center'] )
        image_point, _  =  cv2.projectPoints( np.array([gaze_point]) , self.rotation_vectors[p_id], self.translation_vectors[p_id] , self.camera_matrix , self.dist_coefs )
        image_point = image_point.reshape(-1,2)
        image_point = normalize( image_point[0], self.world_frame_size , flip_y = True)
        image_point = _clamp_norm_point(image_point)
        if p_id == 0:
            eye_center = self.eye0_to_World(p['sphere']['center'])
            gaze_3d = self.eye0_to_World(gaze_point)
        else:
            eye_center = self.eye1_to_World(p['sphere']['center'])
            gaze_3d = self.eye1_to_World(gaze_point)

        normal_3d = np.dot( self.rotation_matricies[p_id], np.array( p['circle_3d']['normal'] ) )

        g = {
                'topic':'gaze',
                'norm_pos':image_point,
                'eye_centers_3d':{p['id']:eye_center.tolist()},
                'gaze_normals_3d':{p['id']:normal_3d.tolist()},
                'gaze_point_3d':gaze_3d.tolist(),
                'confidence':p['confidence'],
                'timestamp':p['timestamp'],
                'base_data':[p]}


        if self.visualizer.window:
            if p_id == 0:
                self.gaze_pts_debug0.append(gaze_3d)
                self.sphere0['center'] = eye_center
                self.sphere0['radius'] = p['sphere']['radius']
            else:
                self.gaze_pts_debug1.append(gaze_3d)
                self.sphere1['center'] = eye_center
                self.sphere1['radius'] = p['sphere']['radius']

        return g

    def _map_binocular(self, p0, p1):

        if '3d' not in p0['method'] or '3d' not in p1['method']:
            return None

        #find the nearest intersection point of the two gaze lines
        #eye ball centers in world coords
        s0_center = self.eye0_to_World( np.array( p0['sphere']['center'] ) )
        s1_center = self.eye1_to_World( np.array( p1['sphere']['center'] ) )
        #eye line of sight in world coords
        s0_normal = np.dot( self.rotation_matricies[0], np.array( p0['circle_3d']['normal'] ) )
        s1_normal = np.dot( self.rotation_matricies[1], np.array( p1['circle_3d']['normal'] ) )

        # See Lech Swirski: "Gaze estimation on glasses-based stereoscopic displays"
        # Chapter: 7.4.2 Cyclopean gaze estimate

        #the cyclop is the avg of both lines of sight
        cyclop_normal = (s0_normal+s1_normal)/2.
        cyclop_center = (s0_center+s1_center)/2.

        # We use it to define a viewing plane.
        gaze_plane = np.cross(cyclop_normal , s1_center-s0_center)
        gaze_plane = gaze_plane/np.linalg.norm(gaze_plane)

        #project lines of sight onto the gaze plane
        s0_norm_on_plane =  s0_normal - np.dot(gaze_plane,s0_normal)*gaze_plane
        s1_norm_on_plane =  s1_normal - np.dot(gaze_plane,s1_normal)*gaze_plane

        #create gaze lines on this plane
        gaze_line0 = [ s0_center, s0_center + s0_norm_on_plane ]
        gaze_line1 = [ s1_center, s1_center + s1_norm_on_plane ]

        #find the intersection of left and right line of sight.
        nearest_intersection_point , intersection_distance = math_helper.nearest_intersection( gaze_line0, gaze_line1 )
        if nearest_intersection_point is not None :
            cyclop_gaze =  nearest_intersection_point-cyclop_center
            self.last_gaze_distance = np.sqrt( cyclop_gaze.dot( cyclop_gaze ) )
            image_point, _  =  cv2.projectPoints( np.array([nearest_intersection_point]) ,  np.array([0.0,0.0,0.0]) ,  np.array([0.0,0.0,0.0]) , self.camera_matrix , self.dist_coefs )
            image_point = image_point.reshape(-1,2)
            image_point = normalize( image_point[0], self.world_frame_size , flip_y = True)
            image_point = _clamp_norm_point(image_point)


        if self.visualizer.window:
            gaze0_3d =  s0_normal * self.last_gaze_distance  + s0_center
            gaze1_3d =  s1_normal * self.last_gaze_distance  + s1_center
            self.gaze_pts_debug0.append(  gaze0_3d)
            self.gaze_pts_debug1.append(  gaze1_3d)
            if nearest_intersection_point is not None:
                self.intersection_points_debug.append( nearest_intersection_point )

            self.sphere0['center'] = s0_center #eye camera coordinates
            self.sphere0['radius'] = p0['sphere']['radius']
            self.sphere1['center'] = s1_center #eye camera coordinates
            self.sphere1['radius'] = p1['sphere']['radius']

        if nearest_intersection_point is None :
            return None


        confidence = min(p0['confidence'],p1['confidence'])
        ts = (p0['timestamp'] + p1['timestamp'])/2.
        g = {   'topic':'gaze',
                'norm_pos':image_point,
                'eye_centers_3d':{0:s0_center.tolist(),1:s1_center.tolist()},
                'gaze_normals_3d':{0:s0_normal.tolist(),1:s1_normal.tolist()},
                'gaze_point_3d':nearest_intersection_point.tolist(),
                'confidence':confidence,
                'timestamp':ts,
                'base_data':[p0,p1]}
        return g

    def gl_display(self):
        self.visualizer.update_window( self.g_pool , self.gaze_pts_debug0 , self.sphere0, self.gaze_pts_debug1, self.sphere1, self.intersection_points_debug )
        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []

    def get_init_dict(self):
       return {'eye_camera_to_world_matrix0':self.eye_camera_to_world_matricies[0].tolist() ,'eye_camera_to_world_matrix1':self.eye_camera_to_world_matricies[1].tolist() ,'cal_ref_points_3d':self.cal_ref_points_3d, 'cal_gaze_points0_3d':self.cal_gaze_points0_3d, 'cal_gaze_points1_3d':self.cal_gaze_points1_3d,  "camera_intrinsics":self.camera_intrinsics}


    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None


    def cleanup(self):
        super().cleanup()
        self.deinit_gui()
        self.visualizer.close_window()
