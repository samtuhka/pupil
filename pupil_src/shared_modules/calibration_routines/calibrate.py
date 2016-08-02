'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2

from methods import undistort_unproject_pts
#logging
import logging
logger = logging.getLogger(__name__)
import statsmodels.api as sm
import math


def calibrate_2d_polynomial(cal_pt_cloud,screen_size=(1,1),threshold = 35, binocular=False):
    """
    we do a simple two pass fitting to a pair of bi-variate polynomials
    return the function to map vector
    """
    # fit once using all avaiable data
    model_n = 7
    if binocular:
        model_n = 13

    cal_pt_cloud = np.array(cal_pt_cloud)

    cx,cy,err_x,err_y = fit_poly_surface(cal_pt_cloud,model_n)
    err_dist,err_mean,err_rms = fit_error_screen(err_x,err_y,screen_size)
    if cal_pt_cloud[err_dist<=threshold].shape[0]: #did not disregard all points..
        # fit again disregarding extreme outliers
        cx,cy,new_err_x,new_err_y = fit_poly_surface(cal_pt_cloud[err_dist<=threshold],model_n)
        map_fn = make_map_function(cx,cy,model_n)
        new_err_dist,new_err_mean,new_err_rms = fit_error_screen(new_err_x,new_err_y,screen_size)

        logger.info('first iteration. root-mean-square residuals: %s, in pixel' %err_rms)
        logger.info('second iteration: ignoring outliers. root-mean-square residuals: %s in pixel'%new_err_rms)

        logger.info('used %i data points out of the full dataset %i: subset is %i percent' \
            %(cal_pt_cloud[err_dist<=threshold].shape[0], cal_pt_cloud.shape[0], \
            100*float(cal_pt_cloud[err_dist<=threshold].shape[0])/cal_pt_cloud.shape[0]))

        return map_fn,err_dist<=threshold,(cx,cy,model_n)

    else: # did disregard all points. The data cannot be represented by the model in a meaningful way:
        map_fn = make_map_function(cx,cy,model_n)
        logger.error('First iteration. root-mean-square residuals: %s in pixel, this is bad!'%err_rms)
        logger.error('The data cannot be represented by the model in a meaningfull way.')
        return map_fn,err_dist<=threshold,(cx,cy,model_n)



def fit_poly_surface(cal_pt_cloud,n=7):
    M = make_model(cal_pt_cloud,n)
    U,w,Vt = np.linalg.svd(M[:,:n],full_matrices=0)
    V = Vt.transpose()
    Ut = U.transpose()
    pseudINV = np.dot(V, np.dot(np.diag(1/w), Ut))
    cx = np.dot(pseudINV, M[:,n])
    cy = np.dot(pseudINV, M[:,n+1])
    # compute model error in world screen units if screen_res specified
    err_x=(np.dot(M[:,:n],cx)-M[:,n])
    err_y=(np.dot(M[:,:n],cy)-M[:,n+1])
    return cx,cy,err_x,err_y

def fit_poly_surface_alternative(cal_pt_cloud,n=7):
    M = make_model(cal_pt_cloud,n)
    rlmX = sm.RLM(M[:,n], M[:,:n]).fit()
    rlmY = sm.RLM(M[:,n+1], M[:,:n]).fit()
    cx = rlmX.params
    cy = rlmY.params
    err_x=(np.dot(M[:,:n],cx)-M[:,n])
    err_y=(np.dot(M[:,:n],cy)-M[:,n+1])
    return cx,cy,err_x,err_y

def fit_error_screen(err_x,err_y,(screen_x,screen_y)):
    err_x *= screen_x/2.
    err_y *= screen_y/2.
    err_dist=np.sqrt(err_x*err_x + err_y*err_y)
    err_mean=np.sum(err_dist)/len(err_dist)
    err_rms=np.sqrt(np.sum(err_dist*err_dist)/len(err_dist))
    return err_dist,err_mean,err_rms

def fit_error_angle(err_x,err_y ) :
    err_x *= 2. * np.pi
    err_y *= 2. * np.pi
    err_dist=np.sqrt(err_x*err_x + err_y*err_y)
    err_mean=np.sum(err_dist)/len(err_dist)
    err_rms=np.sqrt(np.sum(err_dist*err_dist)/len(err_dist))
    return err_dist,err_mean,err_rms

def make_model(cal_pt_cloud,n=7):
    n_points = cal_pt_cloud.shape[0]

    if n==3:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]
        M=np.array([X,Y,Ones,ZX,ZY]).transpose()

    elif n==5:
        X0=cal_pt_cloud[:,0]
        Y0=cal_pt_cloud[:,1]
        X1=cal_pt_cloud[:,2]
        Y1=cal_pt_cloud[:,3]
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,4]
        ZY=cal_pt_cloud[:,5]
        M=np.array([X0,Y0,X1,Y1,Ones,ZX,ZY]).transpose()

    elif n==7:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]
        XX=X*X
        YY=Y*Y
        XY=X*Y
        XXYY=XX*YY
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]
        M=np.array([X,Y,XX,YY,XY,XXYY,Ones,ZX,ZY]).transpose()

    elif n==9:
        X=cal_pt_cloud[:,0]
        Y=cal_pt_cloud[:,1]
        XX=X*X
        YY=Y*Y
        XY=X*Y
        XXYY=XX*YY
        XXY=XX*Y
        YYX=YY*X
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,2]
        ZY=cal_pt_cloud[:,3]
        M=np.array([X,Y,XX,YY,XY,XXYY,XXY,YYX,Ones,ZX,ZY]).transpose()


    elif n==10:
        X1=cal_pt_cloud[:,0]
        Y1=cal_pt_cloud[:,1]
        X2=cal_pt_cloud[:,2]
        Y2=cal_pt_cloud[:,3]
        D = (((X1 - X2)*640)**2 + ((Y1 - Y2)*480)**2)**0.5
        D = 1
        X = ((X1 + X2) * 0.5) / D
        Y = ((Y1 + Y2) * 0.5) / D
        XX = X*X
        YY = Y*Y
        XY = X*Y
        XXX = XX*X
        YYY = YY*Y
        XXY = XX*Y
        YYX = YY*X
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,4]
        ZY=cal_pt_cloud[:,5]
        M=np.array([X, Y, XX, YY, XY,XXX,YYY,XXY,YYX, Ones,ZX,ZY]).transpose()

    elif n==13:
        X0=cal_pt_cloud[:,0]
        Y0=cal_pt_cloud[:,1]
        X1=cal_pt_cloud[:,2]
        Y1=cal_pt_cloud[:,3]
        XX0=X0*X0
        YY0=Y0*Y0
        XY0=X0*Y0
        XXYY0=XX0*YY0
        XX1=X1*X1
        YY1=Y1*Y1
        XY1=X1*Y1
        XXYY1=XX1*YY1
        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,4]
        ZY=cal_pt_cloud[:,5]
        M=np.array([X0,Y0,X1,Y1,XX0,YY0,XY0,XXYY0,XX1,YY1,XY1,XXYY1,Ones,ZX,ZY]).transpose()

    elif n==17:
        X0=cal_pt_cloud[:,0]
        Y0=cal_pt_cloud[:,1]
        X1=cal_pt_cloud[:,2]
        Y1=cal_pt_cloud[:,3]
        XX0=X0*X0
        YY0=Y0*Y0
        XY0=X0*Y0
        XXYY0=XX0*YY0
        XX1=X1*X1
        YY1=Y1*Y1
        XY1=X1*Y1
        XXYY1=XX1*YY1

        X0X1 = X0*X1
        X0Y1 = X0*Y1
        Y0X1 = Y0*X1
        Y0Y1 = Y0*Y1

        Ones=np.ones(n_points)

        ZX=cal_pt_cloud[:,4]
        ZY=cal_pt_cloud[:,5]
        M=np.array([X0,Y0,X1,Y1,XX0,YY0,XY0,XXYY0,XX1,YY1,XY1,XXYY1,X0X1,X0Y1,Y0X1,Y0Y1,Ones,ZX,ZY]).transpose()
    elif n==19:
        X1_0=cal_pt_cloud[:,0]
        Y1_0=cal_pt_cloud[:,1]
        X1_1=cal_pt_cloud[:,2]
        Y1_1=cal_pt_cloud[:,3]
        X2_0=cal_pt_cloud[:,4]
        Y2_0=cal_pt_cloud[:,5]
        X2_1=cal_pt_cloud[:,6]
        Y2_1=cal_pt_cloud[:,7]
        X1X2_0=X1_0*X2_0
        Y1Y2_0=Y1_0*Y2_0
        X1Y1_0=X1_0*Y1_0
        X2Y2_0=X2_0*Y2_0
        X1X2Y1Y2_0=X1X2_0*Y1Y2_0

        X1X2_1=X1_1*X2_1
        Y1Y2_1=Y1_1*Y2_1
        X1Y1_1=X1_1*Y1_1
        X2Y2_1=X2_1*Y2_1
        X1X2Y1Y2_1=X1X2_1*Y1Y2_1

        Ones=np.ones(n_points)
        ZX=cal_pt_cloud[:,8]
        ZY=cal_pt_cloud[:,9]
        M=np.array([X1_0, Y1_0, X1_1, Y1_1, X2_0,Y2_0,X2_1,Y2_1, X1X2_0,Y1Y2_0, X1Y1_0, X2Y2_0, X1X2Y1Y2_0, X1X2_1,Y1Y2_1, X1Y1_1, X2Y2_1,X1X2Y1Y2_1, Ones,ZX,ZY]).transpose()


    else:
        raise Exception("ERROR: Model n needs to be 3, 5, 7 or 9")
    return M




def make_map_function_two_glints(cx, cy, n):
    if n==10:
        def fn((X1, Y1, X2, Y2)):
            D = (((X1 - X2)*640)**2 + ((Y1 - Y2)*480)**2)**0.5
            D = 1
            X = ((X1 + X2) * 0.5) / D
            Y = ((Y1 + Y2) * 0.5) / D
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*X*X*X + cx[6]*Y*Y*Y + cx[7]*X*X*Y + cx[8]*Y*Y*X + cx[9]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*X*X*X + cy[6]*Y*Y*Y + cy[7]*X*X*Y + cy[8]*Y*Y*X + cy[9]
            return x2, y2
    elif n==19:
        def fn((X1_0, Y1_0, X2_0, Y2_0),(X1_1, Y1_1, X2_1, Y2_1)):
            x2 = cx[0]*X1_0 + cx[1]*Y1_0 + cx[2]*X1_1 + cx[3]*Y1_1 + cx[4]*X2_0 + cx[5]*Y2_0 + cx[6]*X2_1 + cx[7]*Y2_1 + cx[8]*X1_0*X2_0 + cx[9]*Y1_0*Y2_0 + cx[10]*X1_0*Y1_0 + cx[11]*X2_0*Y2_0 + cx[12]*X1_0*X2_0*Y1_0*Y2_0 + cx[13]*X1_1*X2_1 + cx[14]*Y1_1*Y2_1 + cx[15]*X1_1*Y1_1 + cx[16]*X2_1*Y2_1 + cx[17]*X1_1*X2_1*Y1_1*Y2_1 + cx[18]
            y2 = cy[0]*X1_0 + cy[1]*Y1_0 + cy[2]*X1_1 + cy[3]*Y1_1 + cy[4]*X2_0 + cy[5]*Y2_0 + cy[6]*X2_1 + cy[7]*Y2_1 + cy[8]*X1_0*X2_0 + cy[9]*Y1_0*Y2_0 + cy[10]*X1_0*Y1_0 + cy[11]*X2_0*Y2_0 + cy[12]*X1_0*X2_0*Y1_0*Y2_0 + cy[13]*X1_1*X2_1 + cy[14]*Y1_1*Y2_1 + cy[15]*X1_1*Y1_1 + cy[16]*X2_1*Y2_1 + cy[17]*X1_1*X2_1*Y1_1*Y2_1 + cy[18]
            return x2, y2
    return fn


def make_map_function(cx,cy,n):
    if n==3:
        def fn((X,Y)):
            x2 = cx[0]*X + cx[1]*Y +cx[2]
            y2 = cy[0]*X + cy[1]*Y +cy[2]
            return x2,y2

    elif n==5:
        def fn((X0,Y0),(X1,Y1)):
            #        X0        Y0        X1        Y1        Ones
            x2 = cx[0]*X0 + cx[1]*Y0 + cx[2]*X1 + cx[3]*Y1 + cx[4]
            y2 = cy[0]*X0 + cy[1]*Y0 + cy[2]*X1 + cy[3]*Y1 + cy[4]
            return x2,y2

    elif n==7:
        def fn((X,Y)):
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X +cx[6]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X +cy[6]
            return x2,y2

    elif n==9:
        def fn((X,Y)):
            #          X         Y         XX         YY         XY         XXYY         XXY         YYX         Ones
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X + cx[6]*Y*X*X + cx[7]*Y*Y*X + cx[8]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X + cy[6]*Y*X*X + cy[7]*Y*Y*X + cy[8]
            return x2,y2


    elif n==10:
        def fn((X,Y)):
            #          X         Y         XX         YY         XY         XXYY         XXY         YYX         Ones
            x2 = cx[0]*X + cx[1]*Y + cx[2]*X*X + cx[3]*Y*Y + cx[4]*X*Y + cx[5]*Y*Y*X*X + cx[6]*Y*X*X + cx[7]*Y*Y*X + cx[8]
            y2 = cy[0]*X + cy[1]*Y + cy[2]*X*X + cy[3]*Y*Y + cy[4]*X*Y + cy[5]*Y*Y*X*X + cy[6]*Y*X*X + cy[7]*Y*Y*X + cy[8]
            return x2,y2

    elif n==13:
        def fn((X0,Y0),(X1,Y1)):
            #        X0        Y0        X1         Y1            XX0        YY0            XY0            XXYY0                XX1            YY1            XY1            XXYY1        Ones
            x2 = cx[0]*X0 + cx[1]*Y0 + cx[2]*X1 + cx[3]*Y1 + cx[4]*X0*X0 + cx[5]*Y0*Y0 + cx[6]*X0*Y0 + cx[7]*X0*X0*Y0*Y0 + cx[8]*X1*X1 + cx[9]*Y1*Y1 + cx[10]*X1*Y1 + cx[11]*X1*X1*Y1*Y1 + cx[12]
            y2 = cy[0]*X0 + cy[1]*Y0 + cy[2]*X1 + cy[3]*Y1 + cy[4]*X0*X0 + cy[5]*Y0*Y0 + cy[6]*X0*Y0 + cy[7]*X0*X0*Y0*Y0 + cy[8]*X1*X1 + cy[9]*Y1*Y1 + cy[10]*X1*Y1 + cy[11]*X1*X1*Y1*Y1 + cy[12]
            return x2,y2

    elif n==17:
        def fn((X0,Y0),(X1,Y1)):
            #        X0        Y0        X1         Y1            XX0        YY0            XY0            XXYY0                XX1            YY1            XY1            XXYY1            X0X1            X0Y1            Y0X1        Y0Y1           Ones
            x2 = cx[0]*X0 + cx[1]*Y0 + cx[2]*X1 + cx[3]*Y1 + cx[4]*X0*X0 + cx[5]*Y0*Y0 + cx[6]*X0*Y0 + cx[7]*X0*X0*Y0*Y0 + cx[8]*X1*X1 + cx[9]*Y1*Y1 + cx[10]*X1*Y1 + cx[11]*X1*X1*Y1*Y1 + cx[12]*X0*X1 + cx[13]*X0*Y1 + cx[14]*Y0*X1 + cx[15]*Y0*Y1 + cx[16]
            y2 = cy[0]*X0 + cy[1]*Y0 + cy[2]*X1 + cy[3]*Y1 + cy[4]*X0*X0 + cy[5]*Y0*Y0 + cy[6]*X0*Y0 + cy[7]*X0*X0*Y0*Y0 + cy[8]*X1*X1 + cy[9]*Y1*Y1 + cy[10]*X1*Y1 + cy[11]*X1*X1*Y1*Y1 + cy[12]*X0*X1 + cy[13]*X0*Y1 + cy[14]*Y0*X1 + cy[15]*Y0*Y1 + cy[16]
            return x2,y2

    elif n==19:
        def fn((X1_0, Y1_0, X2_0, Y2_0),(X1_1, Y1_1, X2_1, Y2_1)):
            x2 = cx[0]*X1_0 + cx[1]*Y1_0 + cx[2]*X1_1 + cx[3]*Y1_1 + cx[4]*X2_0 + cx[5]*Y2_0 + cx[6]*X2_1 + cx[7]*Y2_1 + cx[8]*X1_0*X2_0 + cx[9]*Y1_0*Y2_0 + cx[10]*X1_0*Y1_0 + cx[11]*X2_0*Y2_0 + cx[12]*X1_0*X2_0*Y1_0*Y2_0 + cx[13]*X1_1*X2_1 + cx[14]*Y1_1*Y2_1 + cx[15]*X1_1*Y1_1 + cx[16]*X2_1*Y2_1 + cx[17]*X1_1*X2_1*Y1_1*Y2_1 + cx[18]
            y2 = cy[0]*X1_0 + cy[1]*Y1_0 + cy[2]*X1_1 + cy[3]*Y1_1 + cy[4]*X2_0 + cy[5]*Y2_0 + cy[6]*X2_1 + cy[7]*Y2_1 + cy[8]*X1_0*X2_0 + cy[9]*Y1_0*Y2_0 + cy[10]*X1_0*Y1_0 + cy[11]*X2_0*Y2_0 + cy[12]*X1_0*X2_0*Y1_0*Y2_0 + cy[13]*X1_1*X2_1 + cy[14]*Y1_1*Y2_1 + cy[15]*X1_1*Y1_1 + cy[16]*X2_1*Y2_1 + cy[17]*X1_1*X2_1*Y1_1*Y2_1 + cy[18]
            return x2, y2
    else:
        raise Exception("ERROR: unsopported number of coefficiants.")

    return fn


def closest_matches_binocular(ref_pts, pupil_pts,max_dispersion=1/15.):
    '''
    get pupil positions closest in time to ref points.
    return list of dict with matching ref, pupil0 and pupil1 data triplets.
    '''
    ref = ref_pts

    pupil0 = [p for p in pupil_pts if p['id']==0]
    pupil1 = [p for p in pupil_pts if p['id']==1]

    pupil0_ts = np.array([p['timestamp'] for p in pupil0])
    pupil1_ts = np.array([p['timestamp'] for p in pupil1])


    def find_nearest_idx(array,value):
        idx = np.searchsorted(array, value, side="left")
        try:
            if abs(value - array[idx-1]) < abs(value - array[idx]):
                return idx-1
            else:
                return idx
        except IndexError:
            return idx-1

    matched = []

    if pupil0 and pupil1:
        for r in ref_pts:
            closest_p0_idx = find_nearest_idx(pupil0_ts,r['timestamp'])
            closest_p0 = pupil0[closest_p0_idx]
            closest_p1_idx = find_nearest_idx(pupil1_ts,r['timestamp'])
            closest_p1 = pupil1[closest_p1_idx]

            dispersion = max(closest_p0['timestamp'],closest_p1['timestamp'],r['timestamp']) - min(closest_p0['timestamp'],closest_p1['timestamp'],r['timestamp'])
            if dispersion < max_dispersion:
                matched.append({'ref':r,'pupil':closest_p0, 'pupil1':closest_p1})
            else:
                print "to far."
    return matched


def closest_matches_monocular(ref_pts, pupil_pts,max_dispersion=1/15.):
    '''
    get pupil positions closest in time to ref points.
    return list of dict with matching ref and pupil datum.

    if your data is binocular use:
    pupil0 = [p for p in pupil_pts if p['id']==0]
    pupil1 = [p for p in pupil_pts if p['id']==1]
    to get the desired eye and pass it as pupil_pts
    '''

    ref = ref_pts
    pupil0 = pupil_pts
    pupil0_ts = np.array([p['timestamp'] for p in pupil0])

    def find_nearest_idx(array,value):
        idx = np.searchsorted(array, value, side="left")
        try:
            if abs(value - array[idx-1]) < abs(value - array[idx]):
                return idx-1
            else:
                return idx
        except IndexError:
            return idx-1

    matched = []
    if pupil0:
        for r in ref_pts:
            closest_p0_idx = find_nearest_idx(pupil0_ts,r['timestamp'])
            closest_p0 = pupil0[closest_p0_idx]
            dispersion = max(closest_p0['timestamp'],r['timestamp']) - min(closest_p0['timestamp'],r['timestamp'])
            if dispersion < max_dispersion:
                matched.append({'ref':r,'pupil':closest_p0})
            else:
                pass
    return matched


def preprocess_data(pupil_pts,ref_pts,id_filter=(0,), glints=False):
    '''small utility function to deal with timestamped but uncorrelated data
    input must be lists that contain dicts with at least "timestamp" and "norm_pos" and "id:
    filter id must be (0,) or (1,) or (0,1).
    '''
    assert id_filter in ( (0,),(1,),(0,1) )

    if len(ref_pts)<=2:
        return []

    pupil_pts = [p for p in pupil_pts if p['id'] in id_filter]

    # if filter is set to handle binocular data, e.g. (0,1)
    if id_filter == (0,1):
        if glints:
            return preprocess_data_binocular_glint(pupil_pts, ref_pts)
        else:
            return preprocess_data_binocular(pupil_pts, ref_pts)
    else:
        if glints:
            return preprocess_data_glint(pupil_pts, ref_pts)
        else:
            return preprocess_data_monocular(pupil_pts,ref_pts)

def preprocess_data_monocular(pupil_pts,ref_pts):
    cal_data = []
    cur_ref_pt = ref_pts.pop(0)
    next_ref_pt = ref_pts.pop(0)
    while True:
        matched = []
        while pupil_pts:
            #select all points past the half-way point between current and next ref data sample
            if pupil_pts[0]['timestamp'] <=(cur_ref_pt['timestamp']+next_ref_pt['timestamp'])/2.:
                matched.append(pupil_pts.pop(0))
            else:
                for p_pt in matched:
                    #only use close points
                    if abs(p_pt['timestamp']-cur_ref_pt['timestamp']) <= 1/15.: #assuming 30fps + slack
                        try:
                            data_pt = p_pt["norm_pos"][0], p_pt["norm_pos"][1],cur_ref_pt['norm_pos'][0],cur_ref_pt['norm_pos'][1], cur_ref_pt['screenpos'][0], cur_ref_pt['screenpos'][1], p_pt['timestamp'], p_pt['id']
                        except:
                            data_pt = p_pt["norm_pos"][0], p_pt["norm_pos"][1],cur_ref_pt['norm_pos'][0],cur_ref_pt['norm_pos'][1],p_pt['timestamp'], p_pt['id']
                        cal_data.append(data_pt)
                break
        if ref_pts:
            cur_ref_pt = next_ref_pt
            next_ref_pt = ref_pts.pop(0)
        else:
            break
    return cal_data

def preprocess_data_glint(glint_pupil_pts, ref_pts):
    cal_data =[]
    cur_ref_pt = ref_pts.pop(0)
    next_ref_pt = ref_pts.pop(0)
    while True:
        matched = []
        while glint_pupil_pts:
             #select all points past the half-way point between current and next ref data sample
            if glint_pupil_pts[0]['timestamp'] <=(cur_ref_pt['timestamp']+next_ref_pt['timestamp'])/2.:
                matched.append(glint_pupil_pts.pop(0))
            else:
                for gp_pt in matched:
                    #only use close points
                    if abs(gp_pt['timestamp']-cur_ref_pt['timestamp']) <= 1/15.: #assuming 30fps + slack
                        try:
                            data_pt = gp_pt['x'], gp_pt['y'], gp_pt['x2'], gp_pt['y2'], cur_ref_pt['norm_pos'][0],cur_ref_pt['norm_pos'][1],cur_ref_pt['screenpos'][0], cur_ref_pt['screenpos'][1], gp_pt['timestamp'],  gp_pt['id']
                        except:
                            data_pt = gp_pt['x'], gp_pt['y'],cur_ref_pt['norm_pos'][0],cur_ref_pt['norm_pos'][1], gp_pt['timestamp'], gp_pt['id']
                        cal_data.append(data_pt)
                break
        if ref_pts:
            cur_ref_pt = next_ref_pt
            next_ref_pt = ref_pts.pop(0)
        else:
            break
    return cal_data

def interpol_params(cal_data):
    rlmX = sm.RLM(cal_data[:,3], cal_data[:,:3]).fit()
    rlmY = sm.RLM(cal_data[:,4], cal_data[:,:3]).fit()
    cx = rlmX.params
    cy = rlmY.params
    return (cx, cy, 3)

def preprocess_data_binocular(pupil_pts, ref_pts):
    matches = []

    cur_ref_pt = ref_pts.pop(0)
    next_ref_pt = ref_pts.pop(0)
    while True:
        matched = [[], [], cur_ref_pt]
        while pupil_pts:
            #select all points past the half-way point between current and next ref data sample
            if pupil_pts[0]['timestamp'] <=(cur_ref_pt['timestamp']+next_ref_pt['timestamp'])/2.:
                if abs(pupil_pts[0]['timestamp']-cur_ref_pt['timestamp']) <= 1/15.: #assuming 30fps + slack
                    eye_id = pupil_pts[0]['id']
                    matched[eye_id].append(pupil_pts.pop(0))
                else:
                    pupil_pts.pop(0)
            else:
                matches.append(matched)
                break
        if ref_pts:
            cur_ref_pt = next_ref_pt
            next_ref_pt = ref_pts.pop(0)
        else:
            break

    cal_data = []
    for pupil_pts_0, pupil_pts_1, ref_pt in matches:
        # there must be at least one sample for each eye
        if len(pupil_pts_0) <= 0 or len(pupil_pts_1) <= 0:
            continue

        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:
            try:
                data_pt = p0["norm_pos"][0], p0["norm_pos"][1],p1["norm_pos"][0], p1["norm_pos"][1],ref_pt['norm_pos'][0],ref_pt['norm_pos'][1], cur_ref_pt['screenpos'][0], cur_ref_pt['screenpos'][1], p0['timestamp'], p1['timestamp']
            except:
                data_pt = p0["norm_pos"][0], p0["norm_pos"][1],p1["norm_pos"][0], p1["norm_pos"][1],ref_pt['norm_pos'][0],ref_pt['norm_pos'][1], p0['timestamp'], p1['timestamp']

            cal_data.append(data_pt)

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

    return cal_data

def preprocess_data_binocular_glint(pupil_pts, ref_pts):
    matches = []

    cur_ref_pt = ref_pts.pop(0)
    next_ref_pt = ref_pts.pop(0)
    while True:
        matched = [[], [], cur_ref_pt]
        while pupil_pts:
            #select all points past the half-way point between current and next ref data sample
            if pupil_pts[0]['timestamp'] <=(cur_ref_pt['timestamp']+next_ref_pt['timestamp'])/2.:
                if abs(pupil_pts[0]['timestamp']-cur_ref_pt['timestamp']) <= 1/15.: #assuming 30fps + slack
                    eye_id = pupil_pts[0]['id']
                    matched[eye_id].append(pupil_pts.pop(0))
                else:
                    pupil_pts.pop(0)
            else:
                matches.append(matched)
                break
        if ref_pts:
            cur_ref_pt = next_ref_pt
            next_ref_pt = ref_pts.pop(0)
        else:
            break

    cal_data = []
    for pupil_pts_0, pupil_pts_1, ref_pt in matches:
        # there must be at least one sample for each eye
        if len(pupil_pts_0) <= 0 or len(pupil_pts_1) <= 0:
            continue

        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:
            try:
                data_pt = p0["x"], p0["y"],p1["x"], p1["y"], p0["x2"], p0["y2"],p1["x2"], p1["y2"],ref_pt['norm_pos'][0],ref_pt['norm_pos'][1], cur_ref_pt['screenpos'][0], cur_ref_pt['screenpos'][1], p0['timestamp'], p1['timestamp']
            except:
                data_pt = p0["x"], p0["y"],p1["x"], p1["y"],ref_pt['norm_pos'][0],ref_pt['norm_pos'][1], p0['timestamp'], p1['timestamp']

            cal_data.append(data_pt)
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

    return cal_data


def preprocess_2d_data_monocular(matched_data):
    cal_data = []
    for pair in matched_data:
        ref,pupil = pair['ref'],pair['pupil']
        cal_data.append( (pupil["norm_pos"][0], pupil["norm_pos"][1],ref['norm_pos'][0],ref['norm_pos'][1]) )
    return cal_data

def preprocess_2d_data_binocular(matched_data):
    cal_data = []
    for triplet in matched_data:
        ref,p0,p1 = triplet['ref'],triplet['pupil'],triplet['pupil1']
        data_pt = p0["norm_pos"][0], p0["norm_pos"][1],p1["norm_pos"][0], p1["norm_pos"][1],ref['norm_pos'][0],ref['norm_pos'][1]
        cal_data.append( data_pt )
    return cal_data

def preprocess_3d_data(matched_data, camera_intrinsics ):
    camera_matrix = camera_intrinsics["camera_matrix"]
    dist_coefs = camera_intrinsics["dist_coefs"]

    ref_processed = []
    pupil0_processed = []
    pupil1_processed = []

    is_binocular = len(matched_data[0] ) == 3
    for data_point in matched_data:
        try:
            # taking the pupil normal as line of sight vector
            pupil0 = data_point['pupil']
            gaze_vector0 = np.array(pupil0['circle_3d']['normal'])
            pupil0_processed.append( gaze_vector0 )

            if is_binocular: # we have binocular data
                pupil1 = data_point['pupil1']
                gaze_vector1 = np.array(pupil1['circle_3d']['normal'])
                pupil1_processed.append( gaze_vector1 )

            # projected point uv to normal ray vector of camera
            ref = data_point['ref']
            ref_vector =  undistort_unproject_pts(ref['screen_pos'] , camera_matrix, dist_coefs).tolist()[0]
            ref_vector = ref_vector / np.linalg.norm(ref_vector)
            # assuming a fixed (assumed) distance we get a 3d point in world camera 3d coords.
            ref_processed.append( np.array(ref_vector) )

        except KeyError as e:
            # this pupil data point did not have 3d detected data.
            pass

    return ref_processed,pupil0_processed,pupil1_processed


def find_rigid_transform(A, B):
    A = np.matrix(A)
    B = np.matrix(B)
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return np.array(R), np.array(t).reshape(3)

def calculate_residual_3D_Points( ref_points, gaze_points, eye_to_world_matrix ):

    average_distance = 0.0
    distance_variance = 0.0
    transformed_gaze_points = []

    for p in gaze_points:
        point = np.zeros(4)
        point[:3] = p
        point[3] = 1.0
        point = eye_to_world_matrix.dot(point)
        point = np.squeeze(np.asarray(point))
        transformed_gaze_points.append( point[:3] )

    for(a,b) in zip( ref_points, transformed_gaze_points):
        average_distance += np.linalg.norm(a-b)

    average_distance /= len(ref_points)

    for(a,b) in zip( ref_points, transformed_gaze_points):
        distance_variance += (np.linalg.norm(a-b) - average_distance)**2

    distance_variance /= len(ref_points)

    return average_distance, distance_variance
