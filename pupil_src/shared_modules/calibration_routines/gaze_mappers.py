'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Gaze_Mapping_Plugin
from calibrate import make_map_function, make_map_function_two_glints
from copy import deepcopy
import numpy as np


class Dummy_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Dummy_Gaze_Mapper"""
    def __init__(self, g_pool):
        super(Dummy_Gaze_Mapper, self).__init__(g_pool)

    def update(self,frame,events):
        gaze_pts = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_pts.append({'norm_pos':p['norm_pos'][:],'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {}


class Simple_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Simple_Gaze_Mapper"""
    def __init__(self, g_pool,params):
        super(Simple_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fn = make_map_function(*self.params)

    def update(self,frame,events):
        gaze_pts = []

        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_point = self.map_fn(p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}


    # def map_gaze_offline(self,pupil_positions):
    #     min_confidence = self.g_pool.pupil_confidence_threshold
    #     gaze_pts = deepcopy(pupil_positions)
    #     norm_pos = np.array([p['norm_pos'] for p in gaze_pts])
    #     norm_pos = self.map_fn(norm_pos.T)
    #     for n in range(len(gaze_pts)):
    #         gaze_pts[n]['norm_pos'] = norm_pos[0][n],norm_pos[1][n]
    #         gaze_pts[n]['base'] = [pupil_positions[n]]
    #     gaze_pts = filter(lambda g: g['confidence']> min_confidence,gaze_pts)
    #     return gaze_pts


class Volumetric_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self,g_pool,params):
        super(Volumetric_Gaze_Mapper, self).__init__(g_pool)
        self.params = params

    def update(self,frame,events):
        gaze_pts = []
        raise NotImplementedError()
        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}


class Glint_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Simple_Gaze_Mapper"""
    def __init__(self, g_pool, params, interpol_params):
        super(Glint_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fn = make_map_function(*params)
        self.interpol_params = interpol_params
        self.interpol_map = make_map_function(*interpol_params)


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
                if g['glint_found']:
                    v = g['x'], g['y']
                else:
                    v = self.interpol_map((g['x'], g['y']))
                    v = g['x'] - v[0], g['y'] - v[1]
                gaze_glint_point = self.map_fn(v)
                gaze_pts.append({'norm_pos':gaze_glint_point,'confidence':g['pupil_confidence'],'timestamp':g['timestamp'], 'foundGlint': g['glint_found']})
        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params, 'interpol_params': self.interpol_params}

class Binocular_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params):
        super(Binocular_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fns = (make_map_function(*self.params[0:3]),make_map_function(*self.params[3:6]))

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
    def __init__(self, g_pool,params):
        super(Binocular_Glint_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fns = (make_map_function_two_glints(*self.params[0:3]),make_map_function_two_glints(*self.params[3:6]))

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
                timestamp = max(gaze_0['timestamp'], gaze_1['timestamp'])
                if gaze_0['confidence'] > gaze_1['confidence'] + 0.5:
                     gaze_pts.append(gaze_0)
                elif gaze_1['confidence'] > gaze_0['confidence'] + 0.5:
                    gaze_pts.append(gaze_1)
                else:
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
        return {'params':self.params}
