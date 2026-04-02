import numpy as np
import cv2

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

"""
-----------  pySBA  -----------

MIT License (MIT)
Copyright (c) FALL 2016, Jahdiel Alvarez
Author: Jahdiel Alvarez
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
#%%
class PySBA:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices, pointWeights=None):
        """Intializes all the class attributes and instance variables.
            Write the specifications for each variable:
            cameraArray with shape (n_cameras, 11) contains initial estimates of parameters for all cameras.
                    First 3 components in each row form a rotation vector,
                    next 3 components form a translation vector,
                    then a focal distance and two distortion parameters,
                    then x,y image center coordinates
            points_3d with shape (n_points, 3)
                    contains initial estimates of point coordinates in the world frame.
            camera_ind with shape (n_observations,)
                    contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
            point_ind with shape (n_observations,)
                    contains indices of points (from 0 to n_points - 1) involved in each observation.
            points_2d with shape (n_observations, 2)
                    contains measured 2-D coordinates of points projected on images in each observations.
            pointWeights with shape (n_observations, )
                    contains cost function weights for each observation point.
        """
        self.cameraArray = cameraArray
        self.points3D = points3D
        self.points2D = points2D

        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices
        if pointWeights is None:
            pointWeights = np.full_like(point2DIndices, 1)
        self.pointWeights = pointWeights.reshape((-1, 1))

    """ Utils for converting 3D world points to 2D camera points """
    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, cameraArray):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        # points_proj -= cameraArray[:, 9:] / 1778
        f = cameraArray[:, 6]
        k1 = cameraArray[:, 7]
        k2 = cameraArray[:, 8]
        n = np.sum(points_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        points_proj *= (r * f)[:, np.newaxis]
        points_proj += cameraArray[:, 9:]
        return points_proj


"""
Utils
"""
def convertParams(camParams):
    allParams = np.full((len(camParams), 11), np.NaN)
    for nCam in range(len(camParams)):
        p = camParams[nCam][0]
        f = p['K'][0,0]/2 + p['K'][1,1]/2
        r = -R.from_matrix(p['r']).as_rotvec()
        t = p['t']
        c = p['K'][2,0:2]
        d = p['RDistort']
        allParams[nCam,:] = np.hstack((r,t,f,d,c))
    return allParams

def unconvertParams(camParamVec):
    thisK = np.full((3, 3), 0)
    thisK[0, 0] = camParamVec[6]
    thisK[1,1] = camParamVec[6]
    thisK[2,2] = 1
    thisK[2,:2] = camParamVec[9:]
    r = R.from_rotvec(-camParamVec[:3]).as_matrix()
    t = camParamVec[3:6]
    d = camParamVec[7:9]
    return {'K': thisK, 'R':r, 't':t, 'd':d}

def getCameraArray(file_path,
                    camera_ids=['red_cam', 'yellow_cam', 'green_cam', 'blue_cam'],
                    load_opt_array=False, opt_file_name='opt_cam_array.npy'):
    '''
    Params
    ------
    file_path : path to the camera array file (either init or optimized)
    camera_ids: list of camera names
    load_opt_array : if True, loads a previously saved optimized camera array
        If loading the optimized array, provide the file name.

    Returns
    -------
    camera_array : array of camera parameters; shape (n_camera, n_params)
    cam_array_fields : list of field names for each parameter

    Camera parameters are:
        Extrinsics
        ----------
        A 3D rotation vector that rotates the world coordinate axes into camera coordinate axes; array of floats, shape (3, )
        A 3D translation vector that translates the world origin to the camera origin; array of floats, shape (3, )

        Intrinsics
        ----------
        Focal distance in pixels; float
        Distortion params; array of floats, shape (2, )
        Principal point offsets (x, y); array of ints, shape (2, )

    These are initially estimated empirically (see il_rig_control/arena_alignment/init_cam_extrinsics)
    and OneNote notes (Camera Calibration).

    They are optimized during calibration and can be subsequently updated.
    '''
    n_cams = len(camera_ids)

    if load_opt_array:
        camera_array = np.load(f'{file_path}{opt_file_name}')
    else:
        camera_array = np.full((n_cams, 11), np.NaN)
        for i, cam in enumerate(camera_ids):
            camera_array[i] = np.load(f'{file_path}{cam}_array.npy')

    cam_array_fields = [
                        'rot_1', 'rot_2', 'rot_3',
                        'trans_1', 'trans_2', 'trans_3',
                        'focal dist', 'distort_1', 'distort_2',
                        'pt_x', 'pt_y'
                        ]

    return camera_array, cam_array_fields



''' Cropping Utils '''
def crop_from_com(img, centroid, half_width, crop_size=(320,320)):
    '''
    Crops an image around a given centroid (crop dims defined by half_width)
    and resizes to the specified crop_size.
    '''
    ctr = np.round(centroid).astype(int)
    half_width = np.round(half_width).astype(int)
    img_h, img_w = img.shape
    
    xmin = np.min([np.max([ctr[0] - half_width, 0]), img_w - 1])
    xmax = np.max([np.min([ctr[0] + half_width + 1, img_w]), 1])
    ymin = np.min([np.max([ctr[1] - half_width, 0]), img_h - 1])
    ymax = np.max([np.min([ctr[1] + half_width + 1, img_h]), 1])
    
    crop_img = cv2.resize(img[ymin:ymax, xmin:xmax], crop_size, cv2.INTER_AREA)
    min_ind = np.array([xmin, ymin])
    max_ind = np.array([xmax, ymax])
    crop_scale = crop_size / (max_ind - min_ind)
    
    return crop_img, min_ind, crop_scale


def crop_bird(full_img, body_COM,
                camParams, sba,
                this_w3d=0.25,
                min_px=25,
                this_crop_size=(320,320)):
        '''
        Crops around the full bird or the face given a centroid defined by body_COM
        Default params are for the full bird crop

        this_w3d : float, defines the relative cropping scale
        min_px : int, defines the minimum pixel size to crop
        this_crop_size : tuple of ints, defines the dimensions of the cropped image
        '''
        # params
        nCams = full_img.shape[0]
        
        # get the 3D distance of the bird from each camera to determine cropping scale
        com_reproj = sba.project(np.tile(body_COM, (nCams, 1)), camParams) # get reprojected body centroid location for each camera
        camDist = sba.rotate(np.tile(body_COM, (nCams, 1)), camParams[:,:3]) # rotate to camera coordinates
        camDist = camDist[:, 2] + camParams[:,5] # get z-axis distance ie along optical axis
        camScale = camParams[:, 6] / camDist  # convert to focal length divided by distance
        half_width = camScale * this_w3d
        
        # save the cropped image, min index, and crop scale for each camera
        min_ind = np.full((nCams, 1, 2), np.NaN)
        crop_scale = np.full((nCams, 1, 2), np.NaN)
        crop_img = np.zeros((nCams, this_crop_size[1], this_crop_size[0]), dtype='uint8')
        for nCam in range(nCams):
            thisCom = np.maximum(com_reproj[nCam], 0)
            thisCom[0] = np.minimum(thisCom[0], full_img[nCam].shape[1]) # x limit is shape[1]
            thisCom[1] = np.minimum(thisCom[1], full_img[nCam].shape[0]) # y limit is shape[0]
            thisHalfWidth = np.maximum(half_width[nCam], min_px) # minimum 51px image for body
            crop_img[nCam, :, :], min_ind[nCam], crop_scale[nCam] = crop_from_com(full_img[nCam],
                                                                                    thisCom,
                                                                                    thisHalfWidth,
                                                                                    )

        return crop_img, min_ind, crop_scale