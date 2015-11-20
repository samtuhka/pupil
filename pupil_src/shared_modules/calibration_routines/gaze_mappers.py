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
from pyglui import ui


class Dummy_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Dummy_Gaze_Mapper"""
    def __init__(self, g_pool):
        super(Dummy_Gaze_Mapper, self).__init__(g_pool)

    def update(self,frame,events):
        gaze_pts = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_pts.append({'norm_pos':p['norm_pos'][:],'confidence':p['confidence'],'timestamp':p['timestamp'], 'id': p['id'], 'base':[p]})

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
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'], 'id': p['id'], 'base':[p]})

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

class Bilateral_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params,params_eye0,params_eye1):
        super(Gaze_Mapping_Plugin, self).__init__(g_pool)
        self.params = params
        self.params_eye0 = params_eye0
        self.params_eye1 = params_eye1
        self.multivariate = True
        self.map_fn = make_map_function(*self.params)
        self.map_fn_fallback = []
        self.map_fn_fallback.append(make_map_function(*self.params_eye0))
        self.map_fn_fallback.append(make_map_function(*self.params_eye1))

    def init_gui(self):
        self.menu = ui.Growing_Menu('Binocular Gaze Mapping')
        self.g_pool.sidebar.insert(3,self.menu)
        self.menu.append(ui.Switch('multivariate',self,on_val=True,off_val=False,label='Multivariate Mode'))

    def update(self,frame,events):

        pupil_pts_0 = []
        pupil_pts_1 = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                if p['id'] == 0:
                    pupil_pts_0.append(p)
                else:
                    pupil_pts_1.append(p)

        # try binocular mapping (needs at least 1 pupil position in each list)
        gaze_pts = []
        if len(pupil_pts_0) > 0 and len(pupil_pts_1) > 0:
            gaze_pts = self._map_binocular(pupil_pts_0, pupil_pts_1, self.multivariate)
        # fallback to monocular if something went wrong
        else:
            for p in pupil_pts_0:
                gaze_point = self.map_fn_fallback[0](p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'], 'id': p['id'], 'base':[p]})
            for p in pupil_pts_1:
                gaze_point = self.map_fn_fallback[1](p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'], 'id': p['id'], 'base':[p]})

        events['gaze_positions'] = gaze_pts

    def _map_binocular(self, pupil_pts_0, pupil_pts_1,multivariate=True):
        # maps gaze with binocular mapping
        # requires each list to contain at least one item!
        # returns 1 gaze point at minimum
        gaze_pts = []
        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:
            if multivariate:
                gaze_point = self.map_fn(p0['norm_pos'], p1['norm_pos'])
            else:
                gaze_point_eye0 = self.map_fn_fallback[0](p0['norm_pos'])
                gaze_point_eye1 = self.map_fn_fallback[1](p1['norm_pos'])
                gaze_point = (gaze_point_eye0[0] + gaze_point_eye1[0])/2. , (gaze_point_eye0[1] + gaze_point_eye1[1])/2.
            confidence = (p0['confidence'] + p1['confidence'])/2.
            ts = (p0['timestamp'] + p1['timestamp'])/2.
            gaze_pts.append({'norm_pos':gaze_point,'confidence':confidence,'timestamp':ts, 'id': 2, 'base':[p0, p1]})

            # keep sample with higher timestamp and increase the one with lower timestamp
            if p0['timestamp'] <= p1['timestamp'] and pupil_pts_0:
                p0 = pupil_pts_0.pop(0)
                continue
            elif p1['timestamp'] <= p0['timestamp'] and pupil_pts_1:
                p1 = pupil_pts_1.pop(0)
                continue
            elif pupil_pts_0 and not pupil_pts_1:
                p0 = pupil_pts_0.pop(0)
            elif pupil_pts_1 and not pupil_pts_0:
                p1 = pupil_pts_1.pop(0)
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

class Glint_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Simple_Gaze_Mapper"""
    def __init__(self, g_pool, params, interpolParams):
        super(Glint_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fn = make_map_function_two_glints(*params)
        self.interpol_params = interpolParams
        self.interpol_map = make_map_function(*interpolParams)

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
    def __init__(self, g_pool,params, interpolParams):
        super(Binocular_Glint_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.paramsInterpol = interpolParams
        self.map_fns = (make_map_function_two_glints(*self.params[0:3]),make_map_function_two_glints(*self.params[3:6]))
        self.map_fns_interpol = (make_map_function(*self.paramsInterpol[0:3]),make_map_function(*self.paramsInterpol[3:6]))

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
        self.map_fn = make_map_function_two_glints(*self.params)
        self.map_fn_fallback = []
        self.map_fn_fallback.append(make_map_function_two_glints(*self.params_eye0))
        self.map_fn_fallback.append(make_map_function_two_glints(*self.params_eye1))

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
