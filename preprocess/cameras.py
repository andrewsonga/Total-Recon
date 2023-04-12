import os
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist, euclidean
import numpy as np
import liblzfse

#def read_depth(filepath, height=256, width=192):
def read_depth(filepath, image_size):

    print("image_size: {}".format(image_size))

    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    # keep depth_img the same aspect ratio as the input image_size
    num_elems = depth_img.size
    height = (np.sqrt(num_elems / np.prod(image_size)) * image_size[0]).astype(np.int32)
    width = (np.sqrt(num_elems / np.prod(image_size)) * image_size[1]).astype(np.int32)
    depth_img = depth_img.copy().reshape((height, width))

    return depth_img

#def read_conf(filepath, height=256, width=192):
def read_conf(filepath, image_size):
    
    print("image_size: {}".format(image_size))

    with open(filepath, 'rb') as conf_fh:
        raw_bytes = conf_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        conf_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)

    num_elems = conf_img.size
    height = (np.sqrt(num_elems / np.prod(image_size)) * image_size[0]).astype(np.int32)
    width = (np.sqrt(num_elems / np.prod(image_size)) * image_size[1]).astype(np.int32)
    conf_img = conf_img.copy().reshape((height, width))

    return conf_img

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def read_rtks(metadata_dir, depth_dir, conf_dir, recenter=False):
    # returns the OpenCV format, camera parameters (world2cam_rtk) as required by BANMo

    # 1. read meta data
    with open(os.path.join(metadata_dir, 'metadata')) as f:
        data = f.read()
        
    js = json.loads(data)       # reconstructing the data as a dictionary
    K = js['K']
    src_fps = js['fps']
    cam2world = np.array(js['poses'])

    print("rtk shape inside loaded metadata: {}".format(cam2world.shape))

    # 2. change from quaternions to rotation matrix
    cam2world_quat = cam2world[:, :4]                           # (N, 4)
    cam2world_trans = cam2world[:, 4:]                          # (N, 3)
    cam2world_rot = R.from_quat(cam2world_quat).as_matrix()     # (N, 3, 3)

    if recenter:  
        # 3. globally shift the cam2world s.t. the camera pose for the first frame doesn't lie at 0,0,0
        # shift it s.t. the origin lies at the medium of the scene
        depth_firstframe = read_depth(os.path.join(depth_dir, "00000.depth"), np.array([js['h'], js['w']]))
        conf_firstframe = read_conf(os.path.join(conf_dir, "00000.conf"), np.array([js['h'], js['w']]))

        depth_firstframe[np.isnan(depth_firstframe)] = 4                          # max depth recorded by the LiDAR is 4m 
        conf_firstframe[np.isnan(depth_firstframe)] = 0

        K_mat = np.array([K]).reshape((3, 3)).transpose()
        u, v = np.meshgrid(np.arange(0, depth_firstframe.shape[1]), np.arange(0, depth_firstframe.shape[0]))

        u = u.ravel()
        v = v.ravel()
        depth_firstframe = depth_firstframe.ravel()
        conf_firstframe = conf_firstframe.ravel()
        depth_valid = depth_firstframe[conf_firstframe == 2.]
        u_valid = u[conf_firstframe == 2.]
        v_valid = v[conf_firstframe == 2.]

        pixels_homogen = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis = 0)

        # inverse-project the depth measurement of the first frame into the camera space via intrinsics and depth
        points_3d = np.transpose(np.matmul(np.linalg.inv(K_mat), pixels_homogen) * np.repeat(depth_valid[None, :], 3, axis = 0))      # (N, 3)
        print("isnan inside points_3d: {}".format(np.any(np.isnan(points_3d))))
        point_3d_median = geometric_median(points_3d)       # (3,)

        #print("mean of the point: {}".format(np.mean(points_3d, axis = 0)))
        print("point_3d_median in camera frame: {}".format(point_3d_median))

        # transform median point coordinates from the camera frame to the world frame using the transformation of the first frame
        point_3d_median_world = np.matmul(cam2world_rot[0, :, :], point_3d_median[:, None])[:, 0] + cam2world_trans[0, :].copy()             # (3,)

        print("point_3d_median in world frame: {}".format(point_3d_median_world))

        # global shift (let's just globally shift in the z-direction for a predefined amount - let's say 0.5m)
        #cam2world_trans = cam2world_trans - np.repeat(point_3d_median_world[None, :], cam2world_trans.shape[0], axis = 0)       # (N, 3) - (N, 3)
        #cam2world_trans[:, 2] = cam2world_trans[:, 2] - point_3d_median_world[2]
        #cam2world_trans[:, 2] = cam2world_trans[:, 2] - 0.1

    print("INSIDE CAMERA.PY: STEP 4")
    # 4. change from cam2world (ARKit) to world2cam (BANMo's required root-body poses)
    world2cam_rot = np.transpose(cam2world_rot, (0, 2, 1))                          # (N, 3, 3)
    world2cam_trans = -np.matmul(world2cam_rot, cam2world_trans[..., None])         # (N, 3, 1)

    #global shift (this is wrong, we should do global shift before we invert cam2world)
    #world2cam_trans = world2cam_trans - np.repeat(point_3d_median[None, :, None], world2cam_trans.shape[0] , axis = 0)      # (N, 3, 1) - (N, 3, 1) 

    #world2cam_trans = world2cam_trans      # we've moved scaling the gt cameras and depth maps to inside nnutils/train_utils.py and nnutils/rendering.py
    world2cam_rt = np.concatenate([world2cam_rot, world2cam_trans], axis = -1)      # (N, 3, 4)

    K_compact = np.array([K[0], K[4], K[6], K[7]])
    world2cam_rtk = np.concatenate([world2cam_rt, np.repeat(K_compact[None, None, :], world2cam_rt.shape[0], axis = 0)], axis = 1)          # (N, 4, 4)

    print("INSIDE CAMERA.PY: STEP 5")
    # 5. change from OpenGL format (ARKit) to OpenCV format (COLMAP - as required by BANMo)
    #world2cam_rtk[:, 0, :] = -1. * world2cam_rtk[:, 0, :]
    #world2cam_rtk[:, 1, :] = -1. * world2cam_rtk[:, 1, :]
    world2cam_rtk[:, 1, :] = -1. * world2cam_rtk[:, 1, :]
    world2cam_rtk[:, 2, :] = -1. * world2cam_rtk[:, 2, :]

    return world2cam_rtk            # shape = (N, 4, 4), where N = number of frames recorded inside metadata