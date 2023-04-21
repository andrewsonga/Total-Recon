# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================

# adopted from nerf-pl
import numpy as np
import time
import pdb
import torch
import torch.nn.functional as F
from pytorch3d import transforms

from nnutils.geom_utils import lbs, Kmatinv, mat2K, pinhole_cam, obj_to_cam,\
                               vec_to_sim3, rtmat_invert, rot_angle, mlp_skinning,\
                               bone_transform, skinning, vrender_flo, \
                               gauss_mlp_skinning, diff_flo
from nnutils.loss_utils import elastic_loss, visibility_loss, feat_match_loss,\
                                kp_reproj_loss, compute_pts_exp, compute_pts_exp_objs, kp_reproj, kp_reproj_objs, evaluate_mlp

#############################################################
################ modified by Chonghyuk Song #################
def render_rays_objs(models_objs,
                     embeddings_objs,
                     rays_objs,
                     N_samples=64,
                     use_disp=False,
                     perturb=0,
                     noise_std=1,
                     chunk=1024*32,
                     obj_bounds=None,
                     use_fine=False,
                     img_size=None,
                     progress=None,
                     opts=None,
                     opts_objs=None,
                     render_vis=False,
                    ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        chunk: the chunk size in batched inference

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    if opts.debug_detailed:
        torch.cuda.synchronize()
        start_time = time.time()

    if use_fine: N_samples = N_samples//2 # use half samples to importance sample

    # Decompose the inputs for background

    rays_bkgd = rays_objs[-1]
    near = rays_bkgd['near']
    far = rays_bkgd['far']            # (N_rays, 1)

    rays_d_bkgd = rays_bkgd['rays_d']  # (N_rays, 3)
    N_rays = rays_d_bkgd.shape[0]

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays_d_bkgd.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays_d_bkgd.device)
        z_vals = lower + (upper - lower) * perturb_rand

    if opts.debug_detailed:
        torch.cuda.synchronize()
        print('sampling depth points: %.2f'%(time.time()-start_time))

    # produce 3d points in the canonical space of each object, using the same z_vals for all objects
    rays_d_objs = []
    dir_embedded_objs = []
    embedding_xyz_objs = []
    embedding_xyz_sigmargb_objs = []
    xyz_sampled_objs = []
    for obj_index, rays in enumerate(rays_objs):     
        # Extract models from lists
        #embedding_xyz = embeddings['xyz']
        #embedding_dir = embeddings['dir']
        embedding_xyz = embeddings_objs[obj_index]['xyz']
        embedding_dir = embeddings_objs[obj_index]['dir']
        #############################################################
        ################ modified by Chonghyuk Song #################
        if opts_objs[obj_index].disentangled_nerf:
            embedding_xyz_sigmargb = embeddings_objs[obj_index]['xyz_sigmargb']
        else:
            embedding_xyz_sigmargb = None
        #############################################################
        #############################################################

        # decompose the inputs
        rays_o = rays['rays_o']
        rays_d = rays['rays_d']  # both (N_rays, 3)

        # Embed direction
        rays_d_norm = rays_d / rays_d.norm(2,-1)[:,None]
        dir_embedded = embedding_dir(rays_d_norm) # (N_rays, embed_dir_channels)

        # zvals are not optimized
        # produce points in the root body space
        xyz_sampled = rays_o.unsqueeze(1) + \
                            rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

        rays_d_objs.append(rays_d)
        dir_embedded_objs.append(dir_embedded)
        embedding_xyz_objs.append(embedding_xyz)
        embedding_xyz_sigmargb_objs.append(embedding_xyz_sigmargb)
        xyz_sampled_objs.append(xyz_sampled)

    if opts.debug_detailed:
        torch.cuda.synchronize()
        print('sample depth points + embed dirs: %.2f'%(time.time()-start_time))

    if use_fine: # sample points for fine model
        # output: 
        #  loss:   'img_coarse', 'sil_coarse', 'feat_err', 'proj_err' 
        #               'vis_loss', 'flo/fdp_coarse', 'flo/fdp_valid',  
        #  not loss:   'depth_rnd', 'pts_pred', 'pts_exp'
        with torch.no_grad():
            #_, weights_coarse = inference_deform(xyz_sampled, rays, models, 
            #                  chunk, N_samples,
            #                  N_rays, embedding_xyz, rays_d, noise_std,
            #                  obj_bound, dir_embedded, z_vals,
            #                  img_size, progress,opts,fine_iter=False)
            _, weights_coarse = inference_deform_objs(xyz_sampled_objs, rays_objs, models_objs, 
                            chunk, N_samples,
                            N_rays, embedding_xyz_objs, embedding_xyz_sigmargb_objs, rays_d_objs, noise_std,
                            obj_bounds, dir_embedded_objs, z_vals,
                            img_size, progress,opts, opts_objs, fine_iter=False)

        # reset N_importance
        N_importance = N_samples
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) 
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)
        #xyz_sampled = rays_o.unsqueeze(1) + \
        #                   rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

        xyz_sampled_objs = []
        for obj_index, rays in enumerate(rays_objs):        
            # decompose the inputs
            rays_o = rays['rays_o']
            rays_d = rays['rays_d']  # both (N_rays, 3)

            xyz_sampled = rays_o.unsqueeze(1) + \
                            rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

            xyz_sampled_objs.append(xyz_sampled)

        N_samples = N_samples + N_importance # get back to original # of samples
    
        if opts.debug_detailed:
            torch.cuda.synchronize()
            print('embed dir + sample depth points + sample points for fine models: %.2f'%(time.time()-start_time))

    #result, _ = inference_deform(xyz_sampled, rays, models, 
    #                      chunk, N_samples,
    #                      N_rays, embedding_xyz, rays_d, noise_std,
    #                      obj_bound, dir_embedded, z_vals,
    #                      img_size, progress,opts,render_vis=render_vis)
    result, _ = inference_deform_objs(xyz_sampled_objs, rays_objs, models_objs, 
                          chunk, N_samples,
                          N_rays, embedding_xyz_objs, embedding_xyz_sigmargb_objs, rays_d_objs, noise_std,
                          obj_bounds, dir_embedded_objs, z_vals,
                          img_size, progress,opts, opts_objs, render_vis=render_vis)

    if opts.debug_detailed:
        torch.cuda.synchronize()
        print('embed dir + sample depth points + sample points for fine model + run fine model: %.2f'%(time.time()-start_time))

    return result

def inference_deform_objs(xyz_coarse_sampled_objs, rays_objs, models_objs, chunk, N_samples,
                         N_rays, embedding_xyz_objs, embedding_xyz_sigmargb_objs, rays_d_objs, noise_std,
                         obj_bounds, dir_embedded_objs, z_vals,
                         img_size, progress,opts, opts_objs,fine_iter=True, 
                         render_vis=False):
    """
    fine_iter: whether to render loss-related terms
    render_vis: used for novel view synthesis
    opts: we've received as input the opts for the final object (background), which is ok because this method only references the obj-agnostic settings in opts.
    """
    #####################################################################################
    ############################# modified by Chonghyuk Song ############################
    xyz_input_objs = []
    xyz_coarse_target_objs = []
    xyz_coarse_dentrg_objs = []
    xyz_coarse_frame_objs = []
    env_code_objs = []
    model_coarse_objs = []
    frame_cyc_dis_objs = {}
    frame_rigloss_objs = {}
    frame_disp3d_objs = {}

    clip_bound_objs = []
    if render_vis:
        #clip_bound_objs = []
        vis_pred_objs = []
    else:
        #clip_bound_objs = None
        vis_pred_objs = None
    #####################################################################################
    #####################################################################################

    if opts.debug_detailed:
        torch.cuda.synchronize()
        start_time = time.time()

    is_training = models_objs[-1]['coarse'].training

    for obj_index, rays in enumerate(rays_objs):

        xyz_coarse_sampled = xyz_coarse_sampled_objs[obj_index]
        rays = rays_objs[obj_index]
        models = models_objs[obj_index]
        embedding_xyz = embedding_xyz_objs[obj_index]
        #embedding_xyz_sigmargb = embedding_xyz_sigmargb_objs[obj_index]

        obj_bound = obj_bounds[obj_index]

        xys = rays['xys']

        # root space point correspondence in t2
        if opts.dist_corresp:
            xyz_coarse_target = xyz_coarse_sampled.clone()
            xyz_coarse_dentrg = xyz_coarse_sampled.clone()
        xyz_coarse_frame  = xyz_coarse_sampled.clone()

        xyz_coarse_frame_objs.append(xyz_coarse_frame)

        # free deform
        if 'flowbw' in models.keys():
            model_flowbw = models['flowbw']
            model_flowfw = models['flowfw']
            time_embedded = rays['time_embedded'][:,None]
            xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
            flow_bw = evaluate_mlp(model_flowbw, xyz_coarse_embedded, 
                                chunk=chunk//N_samples, xyz=xyz_coarse_sampled, code=time_embedded)
            xyz_coarse_sampled=xyz_coarse_sampled + flow_bw
        
            if fine_iter:
                # cycle loss (in the joint canonical space)
                xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                                    chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded)
                frame_cyc_dis = (flow_bw+flow_fw).norm(2,-1)
                frame_cyc_dis_objs[obj_index] = frame_cyc_dis
                # rigidity loss
                if opts_objs[obj_index].rig_loss:
                    frame_disp3d = flow_fw.norm(2,-1)
                    frame_disp3d_objs[obj_index] = frame_disp3d

                if "time_embedded_target" in rays.keys():
                    time_embedded_target = rays['time_embedded_target'][:,None]
                    flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                            chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_target)
                    xyz_coarse_target=xyz_coarse_sampled + flow_fw
                
                ###############################################################################
                ########################## modified by Chonghyuk Song #########################
                    xyz_coarse_target_objs.append(xyz_coarse_target)
                ###############################################################################
                ###############################################################################

                if "time_embedded_dentrg" in rays.keys():
                    time_embedded_dentrg = rays['time_embedded_dentrg'][:,None]
                    flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                            chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_dentrg)
                    xyz_coarse_dentrg=xyz_coarse_sampled + flow_fw

                ###############################################################################
                ########################## modified by Chonghyuk Song #########################
                    xyz_coarse_dentrg_objs.append(xyz_coarse_dentrg)
                ###############################################################################
                ###############################################################################

        elif 'bones' in models.keys():

            bones_rst = models['bones_rst']
            bone_rts_fw = rays['bone_rts']
            skin_aux = models['skin_aux']
            rest_pose_code =  models['rest_pose_code']
            rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
            
            if 'nerf_skin' in models.keys():
                # compute delta skinning weights of bs, N, B
                nerf_skin = models['nerf_skin'] 
            else:
                nerf_skin = None
            time_embedded = rays['time_embedded'][:,None]
            # coords after deform
            bones_dfm = bone_transform(bones_rst, bone_rts_fw, is_vec=True)
            skin_backward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                        bones_dfm, time_embedded,  nerf_skin, skin_aux=skin_aux)

            # backward skinning
            xyz_coarse_sampled, bones_dfm = lbs(bones_rst, 
                                                    bone_rts_fw, 
                                                    skin_backward,
                                                    xyz_coarse_sampled,
                                                    )

            #if opts.debug_detailed:
            #    torch.cuda.synchronize()
            #    print('backward skinning: %.2f'%(time.time()-start_time))       # 0.03

            if fine_iter:
                #if opts.dist_corresp:
                skin_forward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                            bones_rst,rest_pose_code,  nerf_skin, skin_aux=skin_aux)

                # cycle loss (in the joint canonical space)
                xyz_coarse_frame_cyc,_ = lbs(bones_rst, bone_rts_fw,
                                skin_forward, xyz_coarse_sampled, backward=False)
                frame_cyc_dis = (xyz_coarse_frame - xyz_coarse_frame_cyc).norm(2,-1)
                frame_cyc_dis_objs[obj_index] = frame_cyc_dis

                #if opts.debug_detailed:
                #    torch.cuda.synchronize()
                #    print('forward skinning (for cycle loss): %.2f'%(time.time()-start_time))       # 0.06 - 0.03 = 0.03
                
                # rigidity loss (not used as optimization objective)
                num_bone = bones_rst.shape[0] 
                bone_fw_reshape = bone_rts_fw.view(-1,num_bone,12)
                bone_trn = bone_fw_reshape[:,:,9:12]
                bone_rot = bone_fw_reshape[:,:,0:9].view(-1,num_bone,3,3)
                frame_rigloss = bone_trn.pow(2).sum(-1)+rot_angle(bone_rot)
                frame_rigloss_objs[obj_index] = frame_rigloss
                
                if opts.dist_corresp and 'bone_rts_target' in rays.keys():
                    bone_rts_target = rays['bone_rts_target']
                    xyz_coarse_target,_ = lbs(bones_rst, bone_rts_target,                       # xyz_coarse_target.shape = (N_rays, N_samples, 3)
                                    skin_forward, xyz_coarse_sampled,backward=False)
                    
                ###############################################################################
                ########################## modified by Chonghyuk Song #########################
                    # == 0:
                    #    print("obj {}: xyz_coarse_sampled.shape: {}".format(obj_index, xyz_coarse_sampled.shape))       # train time: shape = (B=3072, N=128, 3); test time: shape = (24576, 128, 3)
                    #    print("obj {}: xyz_coarse_target.shape: {}".format(obj_index, xyz_coarse_target.shape))         # train time: shape = (B=3072, N=128, 3); test time: shape = (24576, 128, 3)

                    xyz_coarse_target_objs.append(xyz_coarse_target)
                ###############################################################################
                ###############################################################################
                

                if opts.dist_corresp and 'bone_rts_dentrg' in rays.keys():
                    bone_rts_dentrg = rays['bone_rts_dentrg']
                    xyz_coarse_dentrg,_ = lbs(bones_rst, bone_rts_dentrg,                       # xyz_coarse_dentrg.shape = (N_rays, N_samples, 3)
                                    skin_forward, xyz_coarse_sampled,backward=False)
                    
                ###############################################################################
                ###############################################################################
                    xyz_coarse_dentrg_objs.append(xyz_coarse_dentrg)
                ###############################################################################
                ###############################################################################
                
                #if opts.debug_detailed:
                #    torch.cuda.synchronize()
                #    print('rigidity loss: %.2f'%(time.time()-start_time))       # 0.07 - 0.06 = 0.01

        # no deformation
        else:
            if fine_iter:
                
                if opts.dist_corresp:
                    xyz_coarse_target_objs.append(xyz_coarse_target)
                    xyz_coarse_dentrg_objs.append(xyz_coarse_dentrg)

        # nerf shape/rgb
        model_coarse = models['coarse']
        if 'env_code' in rays.keys():
            env_code = rays['env_code']
        else:
            env_code = None

        # set out of bounds weights to zero
        clip_bound = obj_bound
        if render_vis: 
            #clip_bound = obj_bound
            xyz_embedded = embedding_xyz(xyz_coarse_sampled)
            vis_pred = evaluate_mlp(models['nerf_vis'], 
                                xyz_embedded, chunk=chunk)[...,0].sigmoid()

            #if opts.debug_detailed:
            #    torch.cuda.synchronize()
            #    print('compute vis_pred: %.2f'%(time.time()-start_time))
        else:
            #clip_bound = None
            vis_pred = None


        if opts.symm_shape:
            ##TODO set to x-symmetric here
            symm_ratio = 0.5
            xyz_x = xyz_coarse_sampled[...,:1].clone()
            symm_mask = torch.rand_like(xyz_x) < symm_ratio
            xyz_x[symm_mask] = -xyz_x[symm_mask]
            xyz_input = torch.cat([xyz_x, xyz_coarse_sampled[...,1:3]],-1)
        else:
            xyz_input = xyz_coarse_sampled

        #####################################################################################
        ############################# modified by Chonghyuk Song ############################
        xyz_input_objs.append(xyz_input)
        env_code_objs.append(env_code)
        model_coarse_objs.append(model_coarse)

        clip_bound_objs.append(clip_bound)
        if render_vis:
            #clip_bound_objs.append(clip_bound)
            vis_pred_objs.append(vis_pred)
        
    #rgb_coarse, depth_rnd, weights_coarse, vis_coarse = \
    #    inference(model_coarse, embedding_xyz, xyz_input, rays_d,
    #            dir_embedded, z_vals, N_rays, N_samples, chunk, noise_std,
    #            weights_only=False, env_code=env_code, 
    #            clip_bound=clip_bound, vis_pred=vis_pred)
    ###################################################################################
    ########################## modified by Chonghyuk Song #############################
    if opts.use_3dcomposite:
        #manual update (04/11 commit from banmo repo)
        #rgb_coarse, depth_rnd, weights_coarse, weights_coarse_objs, weights_coarse_indiv_objs, vis_coarse, sigmas_objs_norm = \
        #    inference_objs(model_coarse_objs, embedding_xyz_objs, xyz_input_objs, rays_d_objs,
        #            dir_embedded_objs, z_vals, N_rays, N_samples, chunk, noise_std,
        #            weights_only=False, env_code_objs=env_code_objs, 
        #            clip_bound_objs=clip_bound_objs, vis_pred_objs=vis_pred_objs)

        #manual update (04/11 commit from banmo repo)
        rgb_coarse, feat_rnd_objs, depth_rnd, weights_coarse, weights_coarse_objs, weights_coarse_indiv_objs, vis_coarse, sigmas_objs_norm = \
            inference_objs(models_objs, embedding_xyz_objs, embedding_xyz_sigmargb_objs, xyz_input_objs, rays_d_objs,
                    dir_embedded_objs, z_vals, N_rays, N_samples, chunk, noise_std,
                    weights_only=False, env_code_objs=env_code_objs, 
                    clip_bound_objs=clip_bound_objs, vis_pred_objs=vis_pred_objs)
    else:
        rgb_coarse, depth_rnd, weights_coarse, weights_coarse_objs, vis_coarse, sigmas_objs_norm = \
            inference_2dcomposite_objs(model_coarse_objs, embedding_xyz_objs, embedding_xyz_sigmargb_objs, xyz_input_objs, rays_d_objs,
                    dir_embedded_objs, z_vals, N_rays, N_samples, chunk, noise_std,
                    weights_only=False, env_code_objs=env_code_objs, 
                    clip_bound_objs=clip_bound_objs, vis_pred_objs=vis_pred_objs)
    #####################################################################################
    #####################################################################################

    #sil_coarse = weights_coarse[:,:-1].sum(1)
    sil_coarse = weights_coarse[:,:-1].detach().sum(1)
    sil_coarse_objs = [weights_coarse_obj[:,:-1].sum(1) for weights_coarse_obj in weights_coarse_objs]

    result = {'img_coarse': rgb_coarse,
              'depth_rnd': depth_rnd,
              'sil_coarse': sil_coarse,
             }

    # a) when we're "naively" computing the silhouette mask for an individual object using JUST the densities from that object
    '''
    sil_coarse_fgs = 0
    for obj_index, sil_coarse_obj in enumerate(sil_coarse_objs):
        if obj_index < len(sil_coarse_objs) - 1:
            result['sil_coarse_obj{}'.format(obj_index)] = sil_coarse_obj
            sil_coarse_fgs = sil_coarse_fgs + sil_coarse_obj
        
        # when we're using "weights" that don't take into account occlusion
        else:
            #print("sil_coarse - sil_coarse_obj: {}".format(sil_coarse - sil_coarse_obj))
            result['sil_coarse_obj{}'.format(obj_index)] = sil_coarse - sil_coarse_fgs

            # just to visualize the actual output of the background
            #result['sil_coarse_obj{}'.format(obj_index)] = sil_coarse_obj
    '''

    # b) when we're computing the silhouette mask for an object by taking into account occlusion and interactions with other objects by taking into account the densities of all objects during computation of transmittance
    for obj_index, sil_coarse_obj in enumerate(sil_coarse_objs):
        result['sil_coarse_obj{}'.format(obj_index)] = sil_coarse_obj

    if opts.debug_detailed:
        torch.cuda.synchronize()
        print('coarse model inference : %.2f'%(time.time()-start_time))         # 0.12 - 0.07 = 0.05
  
    # temporarily disabling this to do eval composite rendering
    # render visibility scores
    if render_vis:
        result['vis_pred'] = (vis_pred * weights_coarse).sum(-1)
    #if is_training:
    if fine_iter:
        # computing expected surface intersection in canonical space and viser feature matching
        if opts.use_corresp:
            # for flow rendering (for now we can't use opts.use_corresp when doing multi-object rendering)
            #pts_exp = compute_pts_exp(weights_coarse, xyz_coarse_sampled)
            #pts_target = kp_reproj(pts_exp, models, embedding_xyz, rays, 
            #                    to_target=True) # N,1,2
            pts_exp_objs = []
            for obj_index, (weights_coarse_indiv_obj, xyz_coarse_sampled_obj), in enumerate(zip(weights_coarse_indiv_objs, xyz_input_objs)):
                pts_exp_obj = compute_pts_exp(weights_coarse_indiv_obj, xyz_coarse_sampled_obj)
                pts_exp_objs.append(pts_exp_obj)

        # viser feature matching for single-object case
        '''
        if 'feats_at_samp' in rays.keys():
            feats_at_samp = rays['feats_at_samp']
            nerf_feat = models['nerf_feat']
            xyz_coarse_sampled_feat = xyz_coarse_sampled
            weights_coarse_feat = weights_coarse
            pts_pred, pts_exp, feat_err = feat_match_loss(nerf_feat, embedding_xyz, 
                    feats_at_samp, xyz_coarse_sampled_feat, weights_coarse_feat,
                    obj_bound, is_training=is_training)

            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('viser feature matching : %.2f'%(time.time()-start_time))     # 0.14 - 0.12 = 0.02
        '''
        
        # viser feature matching, projection loss for multi-object case
        for obj_index, (rays_obj, models_obj) in enumerate(zip(rays_objs, models_objs)):
            
            if 'feats_at_samp' in rays_obj.keys():
                feats_at_samp = rays_obj['feats_at_samp']
                xys = rays_obj['xys']
                nerf_feat = models_obj['nerf_feat']

                embedding_xyz = embedding_xyz_objs[obj_index]
                obj_bound = obj_bounds[obj_index]
                xyz_coarse_sampled_feat = xyz_input_objs[obj_index]
                weights_coarse_feat = weights_coarse_objs[obj_index]
                
                pts_pred, pts_exp, feat_err = feat_match_loss(nerf_feat, embedding_xyz, 
                    feats_at_samp, xyz_coarse_sampled_feat, weights_coarse_feat,
                    obj_bound, is_training=is_training)

                # 3d-2d projection
                proj_err = kp_reproj_loss(pts_pred, xys, models_obj, embedding_xyz, rays_obj)
                proj_err = proj_err/img_size * 2
                
                result['pts_pred_obj{}'.format(obj_index)] = pts_pred
                result['pts_exp_obj{}'.format(obj_index)] = pts_exp
                result['feat_err_obj{}'.format(obj_index)] = feat_err
                result['proj_err_obj{}'.format(obj_index)] = proj_err

        '''
            # 3d-2d projection
            proj_err = kp_reproj_loss(pts_pred, xys, models, 
                    embedding_xyz, rays)
            
            
            result['pts_pred'] = pts_pred                           # sounds like expected ray surface intersection in the canonical space 
            result['pts_exp']  = pts_exp                            # sounds like 3D surface point in canonical space computed via soft argmax descriptor matching
            result['feat_err'] = feat_err # will be used as loss
            result['proj_err'] = proj_err # will be used as loss
        '''
        
        # computing flow correspondence using warp field and camera matrices (single-object)
        '''
        if opts.dist_corresp and 'rtk_vec_target' in rays.keys():           
            # compute correspondence: root space to target view space
            # RT: root space to camera space
            rtk_vec_target =  rays['rtk_vec_target']
            Rmat = rtk_vec_target[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_target[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_target[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_target = obj_to_cam(xyz_coarse_target, Rmat, Tmat)
            xyz_coarse_target = pinhole_cam(xyz_coarse_target,K)
        
        if opts.dist_corresp and 'rtk_vec_dentrg' in rays.keys():
            # compute correspondence: root space to dentrg view space
            # RT: root space to camera space
            rtk_vec_dentrg =  rays['rtk_vec_dentrg']
            Rmat = rtk_vec_dentrg[:,0:9].view(N_rays,1,3,3)                                                         # shape = (N_rays, 1, 3, 3)
            Tmat = rtk_vec_dentrg[:,9:12].view(N_rays,1,3)                                                          # shape = (N_rays, 1, 3)
            Kinv = rtk_vec_dentrg[:,12:21].view(N_rays,1,3,3)                                                       # shape = (N_rays, 1, 3, 3)
            K = mat2K(Kmatinv(Kinv))                                                                                # shape = (N_rays, 1, 4)

            xyz_coarse_dentrg = obj_to_cam(xyz_coarse_dentrg, Rmat, Tmat)                                           # shape = (N_rays, N_samples, 3)
            xyz_coarse_dentrg = pinhole_cam(xyz_coarse_dentrg,K)
        '''

        # computing flow correspondence using warp field and camera matrices (multi-object)
        if opts.dist_corresp and 'rtk_vec_target' in rays_objs[-1].keys():  # opts.dist_corresp will be true for all opts if we use the flag --dist_corresp and 'rtk_vec_target' will be rays.keys() if is_pair = True, which is true if bs>1
            xyz_coarse_target_camspace_objs = []
            
            for obj_index, rays in enumerate(rays_objs):
                # compute correspondence: root space to target view space
                # RT: root space to camera space
                rtk_vec_target =  rays['rtk_vec_target']
                Rmat = rtk_vec_target[:,0:9].view(N_rays,1,3,3)                                                     # shape = (N_rays, 1, 3, 3)
                Tmat = rtk_vec_target[:,9:12].view(N_rays,1,3)                                                      # shape = (N_rays, 1, 3)
                Kinv = rtk_vec_target[:,12:21].view(N_rays,1,3,3)                                                   # shape = (N_rays, 1, 3, 3)
                K = mat2K(Kmatinv(Kinv))                                                                            # shape = (N_rays, 1, 4)

                xyz_coarse_target_camspace = obj_to_cam(xyz_coarse_target_objs[obj_index], Rmat, Tmat)              # shape = (N_rays, N_samples, 3)
                xyz_coarse_target_camspace_objs.append(xyz_coarse_target_camspace)
            
            # use sigmas_norm_objs to compute composite feature, which in this case are the 3D coordinates of the sampled points in the camera space
            xyz_coarse_target_camspace_objs = torch.stack(xyz_coarse_target_camspace_objs, dim = 0)                             # shape = (num_objs, N_rays, N_samples, 3)
            xyz_coarse_target_camspace = torch.sum(sigmas_objs_norm[..., None] * xyz_coarse_target_camspace_objs, dim = 0)      # shape = (N_rays, N_samples_, 3)

            # project xyz_coarse_target_camspace into the screen space
            # for now let's assume all objects have same Kaug and K, so we'll take K from the final object (bkgd) to do 3d-2d projection
            xyz_coarse_target = pinhole_cam(xyz_coarse_target_camspace, K)
        
        if opts.dist_corresp and 'rtk_vec_dentrg' in rays_objs[-1].keys():
            xyz_coarse_dentrg_camspace_objs = []

            for obj_index, rays in enumerate(rays_objs):
                # compute correspondence: root space to dentrg view space
                # RT: root space to camera space
                rtk_vec_dentrg =  rays['rtk_vec_dentrg']
                Rmat = rtk_vec_dentrg[:,0:9].view(N_rays,1,3,3)                                                         # shape = (N_rays, 1, 3, 3)
                Tmat = rtk_vec_dentrg[:,9:12].view(N_rays,1,3)                                                          # shape = (N_rays, 1, 3)
                Kinv = rtk_vec_dentrg[:,12:21].view(N_rays,1,3,3)                                                       # shape = (N_rays, 1, 3, 3)
                K = mat2K(Kmatinv(Kinv))                                                                                # shape = (N_rays, 1, 4)

                xyz_coarse_dentrg_camspace = obj_to_cam(xyz_coarse_dentrg, Rmat, Tmat)                                          # shape = (N_rays, N_samples, 3)
                xyz_coarse_dentrg_camspace_objs.append(xyz_coarse_dentrg_camspace)

            # use sigmas_norm_objs to compute composite feature, which in this case are the 3D coordinates of the sampled points in the camera space
            xyz_coarse_dentrg_camspace_objs = torch.stack(xyz_coarse_dentrg_camspace_objs, dim = 0)                             # shape = (num_objs, N_rays, N_samples, 3)
            xyz_coarse_dentrg_camspace = torch.sum(sigmas_objs_norm[..., None] * xyz_coarse_dentrg_camspace_objs, dim = 0)      # shape = (N_rays, N_samples_, 3)

            # project xyz_coarse_target_camspace into the screen space
            # for now let's assume all objects have same Kaug and K, so we'll take K from the final object (bkgd) to do 3d-2d projection
            xyz_coarse_dentrg = pinhole_cam(xyz_coarse_dentrg_camspace, K)

        if opts.debug_detailed:
            torch.cuda.synchronize()
            print('compute correspondences : %.2f'%(time.time()-start_time))    # 0.15 - 0.14 = 0.01

        # for eikonal loss computation
        for obj_index, rays_obj in enumerate(rays_objs):
            if 'feats_at_samp' in rays_obj.keys():
                result['pts_exp_vis_obj{}'.format(obj_index)] = pts_exp_objs[obj_index]

        '''
        # raw 3d points for visualization
        result['xyz_camera_vis']   = xyz_coarse_frame 
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            result['xyz_canonical_vis']   = xyz_coarse_sampled
        if 'feats_at_samp' in rays.keys():
            result['pts_exp_vis']   = pts_exp
            result['pts_pred_vis']   = pts_pred
        '''

        # computing 3d cycle consistency loss and rigidity loss for single object
        '''
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            # cycle loss (in the joint canonical space)
            #if opts.dist_corresp:
            result['frame_cyc_dis'] = (frame_cyc_dis * weights_coarse.detach()).sum(-1)
            #else:
            #    pts_exp_reg = pts_exp[:,None].detach()
            #    skin_forward = gauss_mlp_skinning(pts_exp_reg, embedding_xyz, 
            #                bones_rst,rest_pose_code,  nerf_skin, skin_aux=skin_aux)
            #    pts_exp_fw,_ = lbs(bones_rst, bone_rts_fw,
            #                      skin_forward, pts_exp_reg, backward=False)
            #    skin_backward = gauss_mlp_skinning(pts_exp_fw, embedding_xyz, 
            #                bones_dfm, time_embedded,  nerf_skin, skin_aux=skin_aux)
            #    pts_exp_fwbw,_ = lbs(bones_rst, bone_rts_fw,
            #                       skin_backward,pts_exp_fw)
            #    frame_cyc_dis = (pts_exp_fwbw - pts_exp_reg).norm(2,-1)
            #    result['frame_cyc_dis'] = sil_coarse.detach() * frame_cyc_dis[...,-1]
            if 'flowbw' in models.keys():
                #result['frame_rigloss'] =  (frame_disp3d  * weights_coarse.detach()).sum(-1)
                ## only evaluate at with_grad mode
                #if xyz_coarse_frame.requires_grad:
                #    # elastic energy
                #    result['elastic_loss'] = elastic_loss(model_flowbw, embedding_xyz, 
                #                      xyz_coarse_frame, time_embedded)
                pass
            else:
                result['frame_rigloss'] =  (frame_rigloss).mean(-1)
                #for obj_index in frame_rigloss_objs:
                #    result['frame_rigloss{}'.format(obj_index)] =  (frame_rigloss[obj_index]).mean(-1)
            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('compute cycle loss : %.2f'%(time.time()-start_time))     # 0.15 - 0.15 = 0.00
            
            ### script to plot sigmas/weights
            #from matplotlib import pyplot as plt
            #plt.ioff()
            #plt.plot(weights_coarse[weights_coarse.sum(-1)==1][:].T.cpu().numpy(),'*-')
            #plt.savefig('weights.png')
            #plt.cla()
            #plt.plot(sigmas[weights_coarse.sum(-1)==1][:].T.cpu().numpy(),'*-')
            #plt.savefig('sigmas.png')
        '''

        # computing 3d cycle loss and rigidity loss for multi-object version
        for obj_index, (rays_obj, models_obj) in enumerate(zip(rays_objs, models_objs)):

            if 'flowbw' in models_obj.keys() or  'bones' in models_obj.keys():
                result['frame_cyc_dis_obj{}'.format(obj_index)] = (frame_cyc_dis_objs[obj_index] * weights_coarse_objs[obj_index].detach()).sum(-1)
                
                if 'flowbw' in models_obj.keys():
                    xyz_coarse_frame = xyz_coarse_frame_objs[obj_index]
                    model_flowbw = models_obj['flowbw']
                    embedding_xyz = embedding_xyz_objs[obj_index]
                    time_embedded = rays_obj['time_embedded'][:,None]

                    if opts_objs[obj_index].rig_loss:
                        result['frame_rigloss_obj{}'.format(obj_index)] = (frame_disp3d_objs[obj_index] * weights_coarse_objs[obj_index].detach()).sum(-1)
                    # only evaluate at with_grad mode
                    if xyz_coarse_frame.requires_grad:
                        # elastic energy
                        if opts_objs[obj_index].elastic_loss:
                            result['elastic_loss_obj{}'.format(obj_index)] = elastic_loss(model_flowbw, embedding_xyz, 
                                            xyz_coarse_frame, time_embedded)
                else:
                    if opts_objs[obj_index].rig_loss:
                        result['frame_rigloss_obj{}'.format(obj_index)] = (frame_rigloss_objs[obj_index]).mean(-1)

        ###############################################################################
        ########################## modified by Chonghyuk Song #########################
        # seems redundant with the piece of code below
        '''
        if is_training and 'nerf_vis' in models.keys():
            result['vis_loss'] = visibility_loss(models['nerf_vis'], embedding_xyz,
                            xyz_coarse_sampled, vis_coarse, obj_bound, chunk)            

            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('compute visibility loss : %.2f'%(time.time()-start_time))        # 0.18 - 0.15 = 0.03
        '''
        ###############################################################################
        ###############################################################################

        # let's implement vis_loss just the background - since we're only using the vis_pred, clip_bound, and the object bounds from the bkgd, which we assume is going to cover the most space
        if is_training and 'nerf_vis' in models_objs[-1].keys():
            result['vis_loss'] = visibility_loss(models_objs[-1]['nerf_vis'], embedding_xyz_objs[-1],
                            xyz_input_objs[-1], vis_coarse, obj_bounds[-1], chunk)            

        # render flow 
        #if 'rtk_vec_target' in rays.keys():
        if 'rtk_vec_target' in rays_objs[-1].keys():
            if opts.dist_corresp:
                flo_coarse, flo_valid = vrender_flo(weights_coarse, xyz_coarse_target,
                                                    xys, img_size)
            else:
                flo_coarse = diff_flo(pts_target, xys, img_size)
                flo_valid = torch.ones_like(flo_coarse[...,:1])

            result['flo_coarse'] = flo_coarse
            result['flo_valid'] = flo_valid

            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('render flow : %.2f'%(time.time()-start_time))                    # 0.18 - 0.18 = 0.00

        if 'rtk_vec_dentrg' in rays.keys():
            if opts.dist_corresp:
                fdp_coarse, fdp_valid = vrender_flo(weights_coarse, 
                                                    xyz_coarse_dentrg, xys, img_size)
            else:
                fdp_coarse = diff_flo(pts_dentrg, xys, img_size)
                fdp_valid = torch.ones_like(fdp_coarse[...,:1])
            result['fdp_coarse'] = fdp_coarse
            result['fdp_valid'] = fdp_valid

        '''
        if 'nerf_unc' in models.keys():
            # xys: bs,nsample,2
            # t: bs
            nerf_unc = models['nerf_unc']
            ts = rays['ts']
            vid_code = rays['vid_code']

            # change according to K
            xysn = rays['xysn']
            xyt = torch.cat([xysn, ts],-1)
            xyt_embedded = embedding_xyz(xyt)
            xyt_code = torch.cat([xyt_embedded, vid_code],-1)
            unc_pred = nerf_unc(xyt_code)
            #TODO add activation function
            #unc_pred = F.softplus(unc_pred)
            result['unc_pred'] = unc_pred
        '''
        for obj_index, (rays_obj, models_obj) in enumerate(zip(rays_objs, models_objs)):
            if 'nerf_unc' in models_obj.keys():
                # xys: bs,nsample,2
                # t: bs
                nerf_unc = models_obj['nerf_unc']
                ts = rays_obj['ts']
                vid_code = rays_obj['vid_code']

                # change according to K
                xysn = rays_obj['xysn']
                xyt = torch.cat([xysn, ts],-1)
                xyt_embedded = embedding_xyz_objs[obj_index](xyt)
                xyt_code = torch.cat([xyt_embedded, vid_code],-1)
                unc_pred = nerf_unc(xyt_code)
                #TODO add activation function
                #unc_pred = F.softplus(unc_pred)
                result['unc_pred_obj{}'.format(obj_index)] = unc_pred

        #if 'img_at_samp' in rays.keys():
        if 'img_at_samp' in rays_objs[-1].keys():
            # compute other losses
            #img_at_samp = rays['img_at_samp']
            #sil_at_samp = rays['sil_at_samp']
            #vis_at_samp = rays['vis_at_samp']
            #flo_at_samp = rays['flo_at_samp']
            #cfd_at_samp = rays['cfd_at_samp']
            #############################################################
            ################ modified by Chonghyuk Song #################
            #dep_at_samp = rays['dep_at_samp']
            #conf_at_samp = rays['conf_at_samp']
            img_at_samp = rays_objs[-1]['img_at_samp']
            #sil_at_samp = rays_objs[-1]['sil_at_samp']
            #vis_at_samp = rays_objs[-1]['vis_at_samp']
            sil_at_samp_objs = [rays_obj['sil_at_samp'] for rays_obj in rays_objs]
            vis_at_samp_objs = [rays_obj['vis_at_samp'] for rays_obj in rays_objs]
            flo_at_samp = rays_objs[-1]['flo_at_samp']
            cfd_at_samp = rays_objs[-1]['cfd_at_samp']
            dep_at_samp = rays_objs[-1]['dep_at_samp']
            conf_at_samp = rays_objs[-1]['conf_at_samp']

            # depth loss (depth_rnd.shape = [N], dep_at_samp.shape = [N, 1], conf_at_samp = [N, 1])
            # TODO: automate the depth scaling factor
            #dep_scale = 0.2
            dep_loss_samp = (depth_rnd[..., None] - opts.dep_scale * dep_at_samp).pow(2) * (conf_at_samp >= 1.5) * ((0 < dep_at_samp) & (dep_at_samp < 4.0))        # depth confidence values: 0 = low, 1 = medium, 2 = high
            #############################################################
            #############################################################

            # img loss
            #img_loss_samp = (rgb_coarse - img_at_samp).pow(2)                                                      #manual update (04/11 commit from banmo repo)
            img_loss_samp = (rgb_coarse - img_at_samp).pow(2).mean(-1)[..., None]                                   #manual update (04/11 commit from banmo repo)
            
            # sil loss, weight sil loss based on # points
            # single-obj case                    (pre 10/09/22)
            '''
            sil_loss_samp_objs = []
            for obj_index, (sil_at_samp, vis_at_samp) in enumerate(zip(sil_at_samp_objs, vis_at_samp_objs)):
                if is_training and sil_at_samp.sum()>0 and (1-sil_at_samp).sum()>0:
                    pos_wt = vis_at_samp.sum()/   sil_at_samp[vis_at_samp>0].sum()
                    neg_wt = vis_at_samp.sum()/(1-sil_at_samp[vis_at_samp>0]).sum()
                    sil_balance_wt = 0.5*pos_wt*sil_at_samp + 0.5*neg_wt*(1-sil_at_samp)
                else: sil_balance_wt = 1
                #sil_loss_samp = (sil_coarse[...,None] - sil_at_samp).pow(2) * sil_balance_wt
                sil_loss_samp = (sil_coarse_objs[obj_index][...,None] - sil_at_samp).pow(2) * sil_balance_wt
                sil_loss_samp = sil_loss_samp * vis_at_samp
                sil_loss_samp_objs.append(sil_loss_samp)
            '''
            # multi-obj case                    (post 10/09/22)
            # sil loss, weight sil loss based on # points
            # remove from consideration frames with invalid masks (i.e. all entries == 255),
            # in which case, sil_at_samp[sil_at_samp < 255] will be empty, and therefore sil_at_samp.sum() = 0
            # note that we DON'T want to remove from consideration pixels belong to "other object (i.e. pixels == 254)
            sil_loss_samp_objs = []
            for obj_index, (sil_at_samp, vis_at_samp) in enumerate(zip(sil_at_samp_objs, vis_at_samp_objs)):

                sil_at_samp_valid = ((0 < sil_at_samp) & (sil_at_samp < 254)).float()
                if is_training and sil_at_samp_valid.sum()>0 and (1-sil_at_samp_valid).sum()>0:
                    pos_wt = vis_at_samp.sum()/   sil_at_samp_valid[vis_at_samp>0].sum()
                    neg_wt = vis_at_samp.sum()/(1-sil_at_samp_valid[vis_at_samp>0]).sum()
                    sil_balance_wt = 0.5*pos_wt*sil_at_samp_valid + 0.5*neg_wt*(1-sil_at_samp_valid)
                else: sil_balance_wt = 1

                sil_loss_samp = (sil_coarse_objs[obj_index][...,None] - ((0 < sil_at_samp) & (sil_at_samp < 254)).float()).pow(2) * sil_balance_wt
                sil_loss_samp = sil_loss_samp * vis_at_samp
                sil_loss_samp[sil_at_samp >= 254.] *= 0

                sil_loss_samp_objs.append(sil_loss_samp)

            # flo loss, confidence weighting: 30x normalized distance - 0.1x pixel error
            flo_loss_samp = (flo_coarse - flo_at_samp).pow(2).sum(-1)
            # hard-threshold cycle error
            #sil_at_samp_flo = (sil_at_samp>0)\
            #         & (flo_valid==1)
            
            # multi-object case (assuming same Kaug across all objects)     
            # no longer using the "sil_filter" flag
            # sil_at_samp_objs portion of sil_at_samp_flo will be applied in forward_default() in scene.py
            #if opts.sil_filter:                                                    #manual update (04/11 commit from banmo repo)
            #    # taking the sil_at_samp for the final object in list              #manual update (04/11 commit from banmo repo)
            #    sil_at_samp_flo = (sil_at_samp_objs[-1]>0) & (flo_valid==1)        #manual update (04/11 commit from banmo repo)
            #else:                                                                  #manual update (04/11 commit from banmo repo)
            sil_at_samp_flo = (flo_valid==1)
            sil_at_samp_flo[cfd_at_samp==0] = False 
            if sil_at_samp_flo.sum()>0:
                cfd_at_samp = cfd_at_samp / cfd_at_samp[sil_at_samp_flo].mean()
            flo_loss_samp = flo_loss_samp[...,None] * cfd_at_samp
    
            result['img_at_samp']   = img_at_samp
            #result['sil_at_samp']   = sil_at_samp
            #result['vis_at_samp']   = vis_at_samp
            result['dep_at_samp']   = dep_at_samp * opts.dep_scale
            result['conf_at_samp']   = conf_at_samp
            result['sil_at_samp_flo']   = sil_at_samp_flo
            result['flo_at_samp']   = flo_at_samp
            result['img_loss_samp'] = img_loss_samp 
            #result['sil_loss_samp'] = sil_loss_samp
            result['flo_loss_samp'] = flo_loss_samp
            #############################################################
            ################ modified by Chonghyuk Song #################
            result['dep_loss_samp'] = dep_loss_samp
            
            for obj_index, (sil_at_samp, vis_at_samp, sil_loss_samp) in enumerate(zip(sil_at_samp_objs, vis_at_samp_objs, sil_loss_samp_objs)):
                result['sil_at_samp_obj{}'.format(obj_index)] = sil_at_samp
                result['vis_at_samp_obj{}'.format(obj_index)] = vis_at_samp
                result['sil_loss_samp_obj{}'.format(obj_index)] = sil_loss_samp
            #############################################################
            #############################################################

            # density regularization loss for multi-object version (no-overlap regularization)
            if opts_objs[-1].use_ent:
                if len(rays_objs) > 1:
                    entropy = -sigmas_objs_norm * torch.log(sigmas_objs_norm + 1e-10)       # shape = (num_objs, N_rays, N_samples)
                    entropy = torch.mean(torch.sum(entropy, dim = 0))
                    result['entropy_loss_samp'] = entropy                       

        # feature rendering loss
        for obj_index, (rays_obj, feat_rnd_obj) in enumerate(zip(rays_objs, feat_rnd_objs)):
            
            if 'feats_at_samp' in rays_obj.keys():
                feats_at_samp = rays_obj['feats_at_samp']
                feat_rnd_obj = F.normalize(feat_rnd_obj, 2, -1)
                frnd_loss_samp = (feat_rnd_obj - feats_at_samp).pow(2).mean(-1)
                result['frnd_loss_samp_obj{}'.format(obj_index)] = frnd_loss_samp

        # for eikonal loss computation
        for obj_index, rays_obj in enumerate(rays_objs):
            if opts_objs[obj_index].dense_trunc_eikonal_loss:
                # btw, this way of computing the query points only works when there's no linear blend skinning component
                assert(not opts_objs[obj_index].lbs)
                rays_o_obj = rays_obj['rays_o']
                rays_d_obj = rays_obj['rays_d']                                                                         # both (N_rays, 3)

                z_steps_trunc = torch.linspace(0, 1, opts_objs[obj_index].ntrunc, device = rays_d_obj.device)           # (opts.ntrunc)
                z_steps_trunc = z_steps_trunc.expand(N_rays, opts_objs[obj_index].ntrunc)                               # (N_rays, opts.ntrunc)

                z_vals_trunc = opts_objs[obj_index].dep_scale * ((dep_at_samp - opts_objs[obj_index].truncation) * (1 - z_steps_trunc) + (dep_at_samp + opts_objs[obj_index].truncation) * z_steps_trunc)           # (N_rays, opts.ntrunc)
                xyz_trunc_region = rays_o_obj.unsqueeze(1) + rays_d_obj.unsqueeze(1) * z_vals_trunc[..., None]          # (N_rays, opts.ntrunc, 3)
                result['xyz_trunc_region_obj{}'.format(obj_index)] = xyz_trunc_region                                   # (N_rays, opts.ntrunc, 3)

    # result = {'img_coarse': rgb_coarse, 'depth_rnd': depth_rnd, 'sil_coarse': sil_coarse}
    return result, weights_coarse

# composite rendering in 3D space (composite in 3D space -> volume rendering)
#def inference_objs(model_objs, embedding_xyz_objs, xyz_objs, dir_objs, dir_embedded_objs, z_vals,          #manual update (04/11 commit from banmo repo)
def inference_objs(models_objs, embedding_xyz_objs, embedding_xyz_sigmargb_objs, xyz_objs, dir_objs, dir_embedded_objs, z_vals, 
        N_rays, N_samples,chunk, noise_std,
        env_code_objs=None, weights_only=False, clip_bound_objs = None, vis_pred_objs=None):                #manual update (04/11 commit from banmo repo)
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model (coarse or fine)
        embedding_xyz: embedding module for xyz
        xyz_: (N_rays, N_samples_, 3) sampled positions
              N_samples_ is the number of sampled points in each ray;
                         = N_samples for coarse model
                         = N_samples+N_importance for fine model
        dir_: (N_rays, 3) ray directions
        dir_embedded: (N_rays, embed_dir_channels) embedded directions
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        weights_only: do inference on sigma only or not

    Outputs:
        #if weights_only:
        #    weights: (N_rays, N_samples_): weights of each sample
        #else:
        #    rgb_final: (N_rays, 3) the final rgb image
        #    depth_final: (N_rays) depth map
        #    weights: (N_rays, N_samples_): weights of each sample
            rgb_final: (N_rays, 3) the final rgb image
            depth_final: (N_rays) depth map
            weights: (N_rays, N_samples_): weights of each sample
    """
    #N_samples_ = xyz_.shape[1]
    N_samples_ = xyz_objs[-1].shape[1]
    sigmas_objs = []
    rgbs_objs = []
    feat_objs = []

    noise = torch.randn((N_rays, N_samples_), device=dir_embedded_objs[-1].device) * noise_std

    for obj_index, xyz_ in enumerate(xyz_objs):
        #model = model_objs[obj_index]                          #manual update (04/11 commit from banmo repo)         
        models = models_objs[obj_index]                         #manual update (04/11 commit from banmo repo)
        nerf_sdf = models['coarse']                             #manual update (04/11 commit from banmo repo)

        embedding_xyz = embedding_xyz_objs[obj_index]
        embedding_xyz_sigmargb = embedding_xyz_sigmargb_objs[obj_index]

        dir_ = dir_objs[obj_index]
        dir_embedded = dir_embedded_objs[obj_index]

        if env_code_objs is not None:
            env_code = env_code_objs[obj_index]
        else:
            env_code = None
        
        if clip_bound_objs is not None:
            clip_bound = clip_bound_objs[obj_index]
        else:
            clip_bound = None
        
        # TODO: figure out how to deal with vis_pred
        if vis_pred_objs is not None:
            vis_pred = vis_pred_objs[obj_index]
        else:
            vis_pred = None

        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                        # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        chunk_size=4096
        B = xyz_.shape[0]
        xyz_input = xyz_.view(N_rays,N_samples,3)
        #out = evaluate_mlp(model, xyz_input,                                           #manual update (04/11 commit from banmo repo)

        if embedding_xyz_sigmargb is not None:
            embed = embedding_xyz_sigmargb
        else:
            embed = embedding_xyz

        if nerf_sdf.use_dir:
            dir_embedded = dir_embedded.view(N_rays,N_samples,-1)
        else:
            dir_embedded = None

        out = evaluate_mlp(nerf_sdf, xyz_input,                 
                embed_xyz = embed,
                dir_embedded = dir_embedded,
                code=env_code,
                chunk=chunk_size, sigma_only=weights_only).view(B,-1)                   #manual update (04/11 commit from banmo repo)

        #manual update (04/11 commit from banmo repo)
        '''
        if weights_only:
            sdf = -out.view(N_rays, N_samples_)
            sdf = sdf + noise
            ibetas = 1/(model.beta.abs()+1e-9)
            sigmas_obj = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
            sigmas_obj = sigmas_obj * ibetas

            sigmas_objs.append(sigmas_obj)    # (N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs_obj = rgbsigma[..., :3]

            sdf = -rgbsigma[..., 3]
            sdf = sdf + noise
            ibetas = 1/(model.beta.abs()+1e-9)

            sigmas_obj = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
            sigmas_obj = sigmas_obj * ibetas

            sigmas_objs.append(sigmas_obj)                # (N_rays, N_samples_)
            rgbs_objs.append(rgbs_obj)                 # (N_rays, N_samples_, 3)
        '''
        rgbsigma = out.view(N_rays, N_samples_, 4)
        rgbs_obj = rgbsigma[..., :3]
        sdf = -rgbsigma[..., 3]
        sdf = sdf + noise
        #ibetas = 1/(model.beta.abs()+1e-9)                                                                 #manual update (04/11 commit from banmo repo)
        ibetas = 1/(nerf_sdf.beta.abs()+1e-9)                                                               #manual update (04/11 commit from banmo repo)
        sigmas_obj = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
        sigmas_obj = sigmas_obj * ibetas

        if 'nerf_feat' in models.keys():                                                                    #manual update (04/11 commit from banmo repo)
            nerf_feat = models['nerf_feat']                                                                 #manual update (04/11 commit from banmo repo)
            feat_obj = evaluate_mlp(nerf_feat, xyz_input,                                   
                embed_xyz = embedding_xyz,                                                      
                chunk=chunk_size).view(N_rays,N_samples_,-1)                                                #manual update (04/11 commit from banmo repo)
        else:                                                                                               #manual update (04/11 commit from banmo repo)
            feat_obj = torch.zeros_like(rgbs_obj)                                                           #manual update (04/11 commit from banmo repo)
        
        sigmas_objs.append(sigmas_obj)                  # (N_rays, N_samples_)
        rgbs_objs.append(rgbs_obj)                      # (N_rays, N_samples_, 3)
        feat_objs.append(feat_obj)                      # (N_rays, N_samples_, 16)                          #manual update (04/11 commit from banmo repo)

    # modified on 06/06/22
    # apply clip bounds for the features (zeroing out any features that lie out of its own clip bounds)    
    if clip_bound is not None:
        for obj_index, (clip_bound_obj, xyz_obj) in enumerate(zip(clip_bound_objs, xyz_objs)):
            clip_bound_obj = torch.Tensor(clip_bound_obj).to(xyz_obj.device)[None,None]
            oob_obj = (xyz_obj.abs()>clip_bound_obj).sum(-1).view(N_rays,N_samples)==0                      # shape (N_rays, N_samples)
            #rgbs_objs[obj_index] = rgbs_objs[obj_index] * torch.repeat_interleave(oob_obj[..., None], 3, dim = -1)                                # shape (N_rays, N_samples, 3) * (N_rays, N_samples, 1)
            #feat_objs[obj_index] = feat_objs[obj_index] * torch.repeat_interleave(oob_obj[..., None], 16, dim = -1)                                # shape (N_rays, N_samples, 3) * (N_rays, N_samples, 1)
            sigmas_objs[obj_index] = sigmas_objs[obj_index] * oob_obj

    #manual update (04/11 commit from banmo repo)
    '''
    # perform composite rendering                                                                          
    if weights_only:                                                                                       
        sigmas_objs = torch.stack(sigmas_objs, dim = 0)         # (num_objs, N_rays, N_samples_)           
        sigmas = torch.sum(sigmas_objs, dim = 0)                # (N_rays, N_samples_)                     
    else:                                                                                                  
        sigmas_objs = torch.stack(sigmas_objs, dim = 0)                         # (num_objs, N_rays, N_samples_)
        sigmas = torch.sum(sigmas_objs, dim = 0)                                # (N_rays, N_samples_)

        rgbs_objs = torch.stack(rgbs_objs, dim = 0)                             # (num_objs, N_rays, N_samples_, 3)        
        sigmas_objs_norm = torch.div(sigmas_objs + 1e-6, sigmas[None, ...] + 1e-6)     # (num_objs, N_rays, N_samples_)
        rgbs = torch.sum(sigmas_objs_norm[..., None] * rgbs_objs, dim = 0)      # (N_rays, N_samples_, 3)
    '''
    # perform composite rendering
    sigmas_objs = torch.stack(sigmas_objs, dim = 0)                             # (num_objs, N_rays, N_samples_)
    sigmas = torch.sum(sigmas_objs, dim = 0)                                    # (N_rays, N_samples_)

    rgbs_objs = torch.stack(rgbs_objs, dim = 0)                                 # (num_objs, N_rays, N_samples_, 3)
    sigmas_objs_norm = torch.div(sigmas_objs + 1e-6, sigmas[None, ...] + 1e-6)  # (num_objs, N_rays, N_samples_)
    rgbs = torch.sum(sigmas_objs_norm[..., None] * rgbs_objs, dim = 0)          # (N_rays, N_samples_, 3)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    # a hacky way to ensures prob. sum up to 1     
    # while the prob. of last bin does not correspond with the values
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

    '''
    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

    # compute alpha by the formula (3)
    sigmas = sigmas+noise
    #sigmas = F.softplus(sigmas)
    #sigmas = torch.relu(sigmas)
    ibetas = 1/(model.beta.abs()+1e-9)
    #ibetas = 100
    sdf = -sigmas
    sigmas = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
    # alternative: 
    #sigmas = F.sigmoid(-sdf*ibetas)
    sigmas = sigmas * ibetas
    '''

    alphas = 1-torch.exp(-deltas*sigmas) # (N_rays, N_samples_), p_i
    alphas_objs = [1-torch.exp(-deltas*sigmas_obj) for sigmas_obj in sigmas_objs]

    #set out-of-bound and nonvisible alphas to zero
    # basically given the way the code is written, we'll be using clip_bound of the bkgd, which is what we should be doing since bkgd spatially covers all other objects
    if clip_bound is not None:
        clip_bound = torch.Tensor(clip_bound).to(xyz_.device)[None,None]
        oob = (xyz_.abs()>clip_bound).sum(-1).view(N_rays,N_samples)>0
        alphas[oob]=0

        for alphas_obj, clip_bound_obj, xyz_obj in zip(alphas_objs, clip_bound_objs, xyz_objs):
            clip_bound_obj = torch.Tensor(clip_bound_obj).to(xyz_obj.device)[None,None]
            oob_obj = (xyz_obj.abs()>clip_bound_obj).sum(-1).view(N_rays,N_samples)>0
            alphas_obj[oob_obj]=0

    ####################################################################
    ################### modified by Chonghyuk Song #####################
    # basically given the way the code is written, we'll be using vis_pred of the bkgd, which is what we should be doing since bkgd spatially covers all other objects
    '''
    if vis_pred is not None:
        alphas[vis_pred<0.02] = 0
        #alphas[vis_pred<0.3] = 0

        for alphas_obj, vis_pred_obj in zip(alphas_objs, vis_pred_objs):
            alphas_obj[vis_pred_obj < 0.02] = 0
            #alphas_obj[vis_pred_obj < 0.3] = 0
    '''
    ####################################################################
    ####################################################################

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    alpha_prod = torch.cumprod(alphas_shifted, -1)[:, :-1]

    alphas_shifted_objs = [torch.cat([torch.ones_like(alphas_obj[:, :1]), 1-alphas_obj+1e-10], -1) for alphas_obj in alphas_objs]
    alpha_prod_objs = [torch.cumprod(alphas_shifted_obj, -1)[:, :-1] for alphas_shifted_obj in alphas_shifted_objs]

    weights = alphas * alpha_prod # (N_rays, N_samples_)

    weights_objs = [alphas_obj * alpha_prod for alphas_obj in alphas_objs]                                                          # weights computed w. consideration for occlusion between objects
    weights_indiv_objs = [alphas_obj * alphas_prod_obj for alphas_obj, alphas_prod_obj in zip(alphas_objs, alpha_prod_objs)]        # individual objects w.o. considering occlusion

    #weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                 # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    visibility = alpha_prod.detach() # 1 q_0 q_j-1
    
    #if weights_only:                               #manual update (04/11 commit from banmo repo)                     
    #    return weights                             #manual update (04/11 commit from banmo repo)

    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2)   # (N_rays, 3)
    feat_final_objs = [torch.sum(weights_obj.unsqueeze(-1)*feat_obj, -2) for (weights_obj, feat_obj) in zip(weights_objs, feat_objs)]   #manual update (04/11 commit from banmo repo)
    depth_final = torch.sum(weights*z_vals, -1)             # (N_rays)

    ####################################################################
    ################### modified by Chonghyuk Song #####################
    #return rgb_final, depth_final, weights, weights_objs, weights_indiv_objs, visibility, sigmas_objs_norm                             #manual update (04/11 commit from banmo repo)
    return rgb_final, feat_final_objs, depth_final, weights, weights_objs, weights_indiv_objs, visibility, sigmas_objs_norm             #manual update (04/11 commit from banmo repo)
    ####################################################################    
    ####################################################################

# composite rendering in 2D screen space (volume render each object -> composite 2D features in screen space)
def inference_2dcomposite_objs(model_objs, embedding_xyz_objs, xyz_objs, dir_objs, dir_embedded_objs, z_vals, 
        N_rays, N_samples,chunk, noise_std,
        env_code_objs=None, weights_only=False, clip_bound_objs = None, vis_pred_objs=None):
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model (coarse or fine)
        embedding_xyz: embedding module for xyz
        xyz_: (N_rays, N_samples_, 3) sampled positions
              N_samples_ is the number of sampled points in each ray;
                         = N_samples for coarse model
                         = N_samples+N_importance for fine model
        dir_: (N_rays, 3) ray directions
        dir_embedded: (N_rays, embed_dir_channels) embedded directions
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        weights_only: do inference on sigma only or not

    Outputs:
        if weights_only:
            weights: (N_rays, N_samples_): weights of each sample
        else:
            rgb_final: (N_rays, 3) the final rgb image
            depth_final: (N_rays) depth map
            weights: (N_rays, N_samples_): weights of each sample
    """

    #N_samples_ = xyz_.shape[1]
    N_samples_ = xyz_objs[-1].shape[1]
    sigmas_objs = []
    rgbs_objs = []

    noise = torch.randn((N_rays, N_samples_), device=dir_embedded_objs[-1].device) * noise_std

    for obj_index, xyz_ in enumerate(xyz_objs):
        model = model_objs[obj_index]
        embedding_xyz = embedding_xyz_objs[obj_index]
        dir_ = dir_objs[obj_index]
        dir_embedded = dir_embedded_objs[obj_index]

        if env_code_objs is not None:
            env_code = env_code_objs[obj_index]
        else:
            env_code = None
        
        if clip_bound_objs is not None:
            clip_bound = clip_bound_objs[obj_index]
        else:
            clip_bound = None
        
        # TODO: figure out how to deal with vis_pred
        if vis_pred_objs is not None:
            vis_pred = vis_pred_objs[obj_index]
        else:
            vis_pred = None

        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                        # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        chunk_size=4096
        B = xyz_.shape[0]
        xyz_input = xyz_.view(N_rays,N_samples,3)
        out = evaluate_mlp(model, xyz_input, 
                embed_xyz = embedding_xyz,
                dir_embedded = dir_embedded.view(N_rays,N_samples,-1),
                code=env_code,
                chunk=chunk_size, sigma_only=weights_only).view(B,-1)

        if weights_only:
            sdf = -out.view(N_rays, N_samples_)
            sdf = sdf + noise
            ibetas = 1/(model.beta.abs()+1e-9)
            sigmas_obj = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
            sigmas_obj = sigmas_obj * ibetas

            sigmas_objs.append(sigmas_obj)    # (N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs_obj = rgbsigma[..., :3]

            sdf = -rgbsigma[..., 3]
            sdf = sdf + noise
            ibetas = 1/(model.beta.abs()+1e-9)

            sigmas_obj = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
            sigmas_obj = sigmas_obj * ibetas

            sigmas_objs.append(sigmas_obj)                # (N_rays, N_samples_)
            rgbs_objs.append(rgbs_obj)                 # (N_rays, N_samples_, 3)
    
    # perform composite rendering
    if weights_only:
        sigmas_objs = torch.stack(sigmas_objs, dim = 0)         # (num_objs, N_rays, N_samples_)
        sigmas = torch.sum(sigmas_objs, dim = 0)                # (N_rays, N_samples_)
    else:
        sigmas_objs = torch.stack(sigmas_objs, dim = 0)                         # (num_objs, N_rays, N_samples_)
        sigmas = torch.sum(sigmas_objs, dim = 0)                                # (N_rays, N_samples_)

        rgbs_objs = torch.stack(rgbs_objs, dim = 0)                             # (num_objs, N_rays, N_samples_, 3)        
        sigmas_objs_norm = torch.div(sigmas_objs + 1e-6, sigmas[None, ...] + 1e-6)     # (num_objs, N_rays, N_samples_)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    # a hacky way to ensures prob. sum up to 1     
    # while the prob. of last bin does not correspond with the values
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

    '''
    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

    # compute alpha by the formula (3)
    sigmas = sigmas+noise
    #sigmas = F.softplus(sigmas)
    #sigmas = torch.relu(sigmas)
    ibetas = 1/(model.beta.abs()+1e-9)
    #ibetas = 100
    sdf = -sigmas
    sigmas = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
    # alternative: 
    #sigmas = F.sigmoid(-sdf*ibetas)
    sigmas = sigmas * ibetas
    '''

    alphas = 1-torch.exp(-deltas*sigmas) # (N_rays, N_samples_), p_i
    alphas_objs = [1-torch.exp(-deltas*sigmas_obj) for sigmas_obj in sigmas_objs]

    #set out-of-bound and nonvisible alphas to zero
    # basically given the way the code is written, we'll be using clip_bound of the bkgd, which is what we should be doing since bkgd spatially covers all other objects
    if clip_bound is not None:
        clip_bound = torch.Tensor(clip_bound).to(xyz_.device)[None,None]
        oob = (xyz_.abs()>clip_bound).sum(-1).view(N_rays,N_samples)>0
        alphas[oob]=0

        for alphas_obj, clip_bound_obj, xyz_obj in zip(alphas_objs, clip_bound_objs, xyz_objs):
            clip_bound_obj = torch.Tensor(clip_bound_obj).to(xyz_obj.device)[None,None]
            oob_obj = (xyz_obj.abs()>clip_bound_obj).sum(-1).view(N_rays,N_samples)>0
            alphas_obj[oob_obj]=0

    ####################################################################
    ################### modified by Chonghyuk Song #####################
    # basically given the way the code is written, we'll be using vis_pred of the bkgd, which is what we should be doing since bkgd spatially covers all other objects
    if vis_pred is not None:
        alphas[vis_pred<0.5] = 0

        for alphas_obj, vis_pred_obj in zip(alphas_objs, vis_pred_objs):
            alphas_obj[vis_pred_obj < 0.5] = 0
    ####################################################################
    ####################################################################

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    alpha_prod = torch.cumprod(alphas_shifted, -1)[:, :-1]

    alphas_shifted_objs = [torch.cat([torch.ones_like(alphas_obj[:, :1]), 1-alphas_obj+1e-10], -1) for alphas_obj in alphas_objs]
    alpha_prod_objs = [torch.cumprod(alphas_shifted_obj, -1)[:, :-1] for alphas_shifted_obj in alphas_shifted_objs]

    weights = alphas * alpha_prod # (N_rays, N_samples_)
    weights_objs = [alphas_obj * alphas_prod_obj for alphas_obj, alphas_prod_obj in zip(alphas_objs, alpha_prod_objs)]

    weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                 # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    visibility = alpha_prod.detach() # 1 q_0 q_j-1
    if weights_only:
        return weights

    # compute final weighted outputs
    rgb_final_objs = [torch.sum(weights_obj.unsqueeze(-1)*rgbs_obj, -2) for weights_obj, rgbs_obj in zip(weights_objs, rgbs_objs)]
    depth_final_objs = [torch.sum(weights_obj*z_vals, -1) for weights_obj in weights_objs]
    
    sil_objs = [weights_obj[:,:-1].sum(1) for weights_obj in weights_objs]                                   # need to modify the background silhouette
    sil_objs[-1] = sil_objs[-1] - torch.sum(torch.stack(sil_objs[:-1], dim = 0), dim = 0)

    rgb_final = torch.sum(torch.stack([rgb_final_obj * sil_obj[..., None] for rgb_final_obj, sil_obj in zip(rgb_final_objs, sil_objs)], dim = 0), dim = 0)
    depth_final = torch.sum(torch.stack([depth_final_obj * sil_obj for depth_final_obj, sil_obj in zip(depth_final_objs, sil_objs)], dim = 0), dim = 0)

    ####################################################################
    ################### modified by Chonghyuk Song #####################
    return rgb_final, depth_final, weights, weights_objs, visibility, sigmas_objs_norm
    ####################################################################
    ####################################################################

#############################################################
#############################################################

def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                chunk=1024*32,
                obj_bound=None,
                use_fine=False,
                img_size=None,
                progress=None,
                opts=None,
                render_vis=False,
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        chunk: the chunk size in batched inference

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    if opts.debug_detailed:
        torch.cuda.synchronize()
        start_time = time.time()

    if use_fine: N_samples = N_samples//2 # use half samples to importance sample

    # Extract models from lists
    embedding_xyz = embeddings['xyz']
    embedding_dir = embeddings['dir']

    # Decompose the inputs
    rays_o = rays['rays_o']
    rays_d = rays['rays_d']  # both (N_rays, 3)
    near = rays['near']
    far = rays['far']  # both (N_rays, 1)
    N_rays = rays_d.shape[0]

    # Embed direction
    rays_d_norm = rays_d / rays_d.norm(2,-1)[:,None]
    dir_embedded = embedding_dir(rays_d_norm) # (N_rays, embed_dir_channels)

    if opts.debug_detailed:
        torch.cuda.synchronize()
        print('embed direction: %.2f'%(time.time()-start_time))

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays_d.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays_d.device)
        z_vals = lower + (upper - lower) * perturb_rand

    # zvals are not optimized
    # produce points in the root body space
    xyz_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    if opts.debug_detailed:
        torch.cuda.synchronize()
        print('embed dir + sample depth points: %.2f'%(time.time()-start_time))

    if use_fine: # sample points for fine model
        # output: 
        #  loss:   'img_coarse', 'sil_coarse', 'feat_err', 'proj_err' 
        #               'vis_loss', 'flo/fdp_coarse', 'flo/fdp_valid',  
        #  not loss:   'depth_rnd', 'pts_pred', 'pts_exp'
        with torch.no_grad():
            _, weights_coarse = inference_deform(xyz_sampled, rays, models, 
                              chunk, N_samples,
                              N_rays, embedding_xyz, rays_d, noise_std,
                              obj_bound, dir_embedded, z_vals,
                              img_size, progress,opts,fine_iter=False)

        # reset N_importance
        N_importance = N_samples
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) 
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

        N_samples = N_samples + N_importance # get back to original # of samples
    
        if opts.debug_detailed:
            torch.cuda.synchronize()
            print('embed dir + sample depth points + sample points for fine model: %.2f'%(time.time()-start_time))

    result, _ = inference_deform(xyz_sampled, rays, models, 
                          chunk, N_samples,
                          N_rays, embedding_xyz, rays_d, noise_std,
                          obj_bound, dir_embedded, z_vals,
                          img_size, progress,opts,render_vis=render_vis)

    if opts.debug_detailed:
        torch.cuda.synchronize()
        print('embed dir + sample depth points + sample points for fine model + run fine model: %.2f'%(time.time()-start_time))

    return result
    
def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, 
        N_rays, N_samples,chunk, noise_std,
        env_code=None, weights_only=False, clip_bound = None, vis_pred=None):
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model (coarse or fine)
        embedding_xyz: embedding module for xyz
        xyz_: (N_rays, N_samples_, 3) sampled positions
              N_samples_ is the number of sampled points in each ray;
                         = N_samples for coarse model
                         = N_samples+N_importance for fine model
        dir_: (N_rays, 3) ray directions
        dir_embedded: (N_rays, embed_dir_channels) embedded directions
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        weights_only: do inference on sigma only or not

    Outputs:
        if weights_only:
            weights: (N_rays, N_samples_): weights of each sample
        else:
            rgb_final: (N_rays, 3) the final rgb image
            depth_final: (N_rays) depth map
            weights: (N_rays, N_samples_): weights of each sample
    """
    N_samples_ = xyz_.shape[1]
    # Embed directions
    xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
    if not weights_only:
        dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                       # (N_rays*N_samples_, embed_dir_channels)

    # Perform model inference to get rgb and raw sigma
    chunk_size=4096
    B = xyz_.shape[0]
    xyz_input = xyz_.view(N_rays,N_samples,3)
    out = evaluate_mlp(model, xyz_input, 
            embed_xyz = embedding_xyz,
            dir_embedded = dir_embedded.view(N_rays,N_samples,-1),
            code=env_code,
            chunk=chunk_size, sigma_only=weights_only).view(B,-1)

    if weights_only:
        sigmas = out.view(N_rays, N_samples_)
    else:
        rgbsigma = out.view(N_rays, N_samples_, 4)
        rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
        sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    # a hacky way to ensures prob. sum up to 1     
    # while the prob. of last bin does not correspond with the values
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

    # compute alpha by the formula (3)
    sigmas = sigmas+noise
    #sigmas = F.softplus(sigmas)
    #sigmas = torch.relu(sigmas)
    ibetas = 1/(model.beta.abs()+1e-9)
    #ibetas = 100
    sdf = -sigmas
    sigmas = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
    # alternative: 
    #sigmas = F.sigmoid(-sdf*ibetas)
    sigmas = sigmas * ibetas

    alphas = 1-torch.exp(-deltas*sigmas) # (N_rays, N_samples_), p_i

    #set out-of-bound and nonvisible alphas to zero
    if clip_bound is not None:
        clip_bound = torch.Tensor(clip_bound).to(xyz_.device)[None,None]
        oob = (xyz_.abs()>clip_bound).sum(-1).view(N_rays,N_samples)>0
        alphas[oob]=0

    ####################################################################
    ################### modified by Chonghyuk Song #####################
    if vis_pred is not None:
        alphas[vis_pred<0.5] = 0
    ####################################################################
    ####################################################################

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    alpha_prod = torch.cumprod(alphas_shifted, -1)[:, :-1]
    weights = alphas * alpha_prod # (N_rays, N_samples_)
    weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                 # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    visibility = alpha_prod.detach() # 1 q_0 q_j-1
    if weights_only:
        return weights

    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
    depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

    return rgb_final, depth_final, weights, visibility
    
def inference_deform(xyz_coarse_sampled, rays, models, chunk, N_samples,
                         N_rays, embedding_xyz, rays_d, noise_std,
                         obj_bound, dir_embedded, z_vals,
                         img_size, progress,opts, fine_iter=True, 
                         render_vis=False):
    """
    fine_iter: whether to render loss-related terms
    render_vis: used for novel view synthesis
    """

    if opts.debug_detailed:
        torch.cuda.synchronize()
        start_time = time.time()

    is_training = models['coarse'].training
    xys = rays['xys']

    # root space point correspondence in t2
    if opts.dist_corresp:
        xyz_coarse_target = xyz_coarse_sampled.clone()
        xyz_coarse_dentrg = xyz_coarse_sampled.clone()
    xyz_coarse_frame  = xyz_coarse_sampled.clone()

    # free deform
    if 'flowbw' in models.keys():
        model_flowbw = models['flowbw']
        model_flowfw = models['flowfw']
        time_embedded = rays['time_embedded'][:,None]
        xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
        flow_bw = evaluate_mlp(model_flowbw, xyz_coarse_embedded, 
                             chunk=chunk//N_samples, xyz=xyz_coarse_sampled, code=time_embedded)
        xyz_coarse_sampled=xyz_coarse_sampled + flow_bw
       
        if fine_iter:
            # cycle loss (in the joint canonical space)
            xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
            flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                                  chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded)
            frame_cyc_dis = (flow_bw+flow_fw).norm(2,-1)
            # rigidity loss
            frame_disp3d = flow_fw.norm(2,-1)

            if "time_embedded_target" in rays.keys():
                time_embedded_target = rays['time_embedded_target'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_target)
                xyz_coarse_target=xyz_coarse_sampled + flow_fw
            
            if "time_embedded_dentrg" in rays.keys():
                time_embedded_dentrg = rays['time_embedded_dentrg'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_dentrg)
                xyz_coarse_dentrg=xyz_coarse_sampled + flow_fw


    elif 'bones' in models.keys():
        bones_rst = models['bones_rst']
        bone_rts_fw = rays['bone_rts']
        skin_aux = models['skin_aux']
        rest_pose_code =  models['rest_pose_code']
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
        
        if 'nerf_skin' in models.keys():
            # compute delta skinning weights of bs, N, B
            nerf_skin = models['nerf_skin'] 
        else:
            nerf_skin = None
        time_embedded = rays['time_embedded'][:,None]
        # coords after deform
        bones_dfm = bone_transform(bones_rst, bone_rts_fw, is_vec=True)
        skin_backward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                    bones_dfm, time_embedded,  nerf_skin, skin_aux=skin_aux)

        # backward skinning
        xyz_coarse_sampled, bones_dfm = lbs(bones_rst, 
                                                  bone_rts_fw, 
                                                  skin_backward,
                                                  xyz_coarse_sampled,
                                                  )

        if opts.debug_detailed:
            torch.cuda.synchronize()
            print('backward skinning: %.2f'%(time.time()-start_time))       # 0.03

        if fine_iter:
            #if opts.dist_corresp:
            skin_forward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                        bones_rst,rest_pose_code,  nerf_skin, skin_aux=skin_aux)

            # cycle loss (in the joint canonical space)
            xyz_coarse_frame_cyc,_ = lbs(bones_rst, bone_rts_fw,
                              skin_forward, xyz_coarse_sampled, backward=False)
            frame_cyc_dis = (xyz_coarse_frame - xyz_coarse_frame_cyc).norm(2,-1)

            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('forward skinning (for cycle loss): %.2f'%(time.time()-start_time))       # 0.06 - 0.03 = 0.03
            
            # rigidity loss (not used as optimization objective)
            num_bone = bones_rst.shape[0] 
            bone_fw_reshape = bone_rts_fw.view(-1,num_bone,12)
            bone_trn = bone_fw_reshape[:,:,9:12]
            bone_rot = bone_fw_reshape[:,:,0:9].view(-1,num_bone,3,3)
            frame_rigloss = bone_trn.pow(2).sum(-1)+rot_angle(bone_rot)
            
            if opts.dist_corresp and 'bone_rts_target' in rays.keys():
                bone_rts_target = rays['bone_rts_target']
                xyz_coarse_target,_ = lbs(bones_rst, bone_rts_target, 
                                   skin_forward, xyz_coarse_sampled,backward=False)
            if opts.dist_corresp and 'bone_rts_dentrg' in rays.keys():
                bone_rts_dentrg = rays['bone_rts_dentrg']
                xyz_coarse_dentrg,_ = lbs(bones_rst, bone_rts_dentrg, 
                                   skin_forward, xyz_coarse_sampled,backward=False)
            
            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('rigidity loss: %.2f'%(time.time()-start_time))       # 0.07 - 0.06 = 0.01

    # nerf shape/rgb
    model_coarse = models['coarse']
    if 'env_code' in rays.keys():
        env_code = rays['env_code']
    else:
        env_code = None

    # set out of bounds weights to zero
    if render_vis: 
        clip_bound = obj_bound
        xyz_embedded = embedding_xyz(xyz_coarse_sampled)
        vis_pred = evaluate_mlp(models['nerf_vis'], 
                               xyz_embedded, chunk=chunk)[...,0].sigmoid()

        if opts.debug_detailed:
            torch.cuda.synchronize()
            print('compute vis_pred: %.2f'%(time.time()-start_time))
    else:
        clip_bound = None
        vis_pred = None


    if opts.symm_shape:
        ##TODO set to x-symmetric here
        symm_ratio = 0.5
        xyz_x = xyz_coarse_sampled[...,:1].clone()
        symm_mask = torch.rand_like(xyz_x) < symm_ratio
        xyz_x[symm_mask] = -xyz_x[symm_mask]
        xyz_input = torch.cat([xyz_x, xyz_coarse_sampled[...,1:3]],-1)
    else:
        xyz_input = xyz_coarse_sampled

    rgb_coarse, depth_rnd, weights_coarse, vis_coarse = \
        inference(model_coarse, embedding_xyz, xyz_input, rays_d,
                dir_embedded, z_vals, N_rays, N_samples, chunk, noise_std,
                weights_only=False, env_code=env_code, 
                clip_bound=clip_bound, vis_pred=vis_pred)
    sil_coarse =  weights_coarse[:,:-1].sum(1)
    result = {'img_coarse': rgb_coarse,
              'depth_rnd': depth_rnd,
              'sil_coarse': sil_coarse,
             }
    if opts.debug_detailed:
        torch.cuda.synchronize()
        print('coarse model inference : %.2f'%(time.time()-start_time))         # 0.12 - 0.07 = 0.05
  
    # render visibility scores
    if render_vis:
        result['vis_pred'] = (vis_pred * weights_coarse).sum(-1)

    if fine_iter:
        if opts.use_corresp:
            # for flow rendering
            pts_exp = compute_pts_exp(weights_coarse, xyz_coarse_sampled)
            pts_target = kp_reproj(pts_exp, models, embedding_xyz, rays, 
                                to_target=True) # N,1,2
        # viser feature matching
        if 'feats_at_samp' in rays.keys():
            feats_at_samp = rays['feats_at_samp']
            nerf_feat = models['nerf_feat']
            xyz_coarse_sampled_feat = xyz_coarse_sampled
            weights_coarse_feat = weights_coarse
            pts_pred, pts_exp, feat_err = feat_match_loss(nerf_feat, embedding_xyz,
                       feats_at_samp, xyz_coarse_sampled_feat, weights_coarse_feat,
                       obj_bound, is_training=is_training)

            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('viser feature matching : %.2f'%(time.time()-start_time))     # 0.14 - 0.12 = 0.02

            # 3d-2d projection
            proj_err = kp_reproj_loss(pts_pred, xys, models, 
                    embedding_xyz, rays)
            proj_err = proj_err/img_size * 2
            
            result['pts_pred'] = pts_pred
            result['pts_exp']  = pts_exp
            result['feat_err'] = feat_err # will be used as loss
            result['proj_err'] = proj_err # will be used as loss

        if opts.dist_corresp and 'rtk_vec_target' in rays.keys():
            # compute correspondence: root space to target view space
            # RT: root space to camera space
            rtk_vec_target =  rays['rtk_vec_target']
            Rmat = rtk_vec_target[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_target[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_target[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_target = obj_to_cam(xyz_coarse_target, Rmat, Tmat) 
            xyz_coarse_target = pinhole_cam(xyz_coarse_target,K)

        if opts.dist_corresp and 'rtk_vec_dentrg' in rays.keys():
            # compute correspondence: root space to dentrg view space
            # RT: root space to camera space
            rtk_vec_dentrg =  rays['rtk_vec_dentrg']
            Rmat = rtk_vec_dentrg[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_dentrg[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_dentrg[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_dentrg = obj_to_cam(xyz_coarse_dentrg, Rmat, Tmat) 
            xyz_coarse_dentrg = pinhole_cam(xyz_coarse_dentrg,K)
        
        if opts.debug_detailed:
            torch.cuda.synchronize()
            print('compute correspondences : %.2f'%(time.time()-start_time))    # 0.15 - 0.14 = 0.01

        # raw 3d points for visualization
        result['xyz_camera_vis']   = xyz_coarse_frame 
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            result['xyz_canonical_vis']   = xyz_coarse_sampled
        if 'feats_at_samp' in rays.keys():
            result['pts_exp_vis']   = pts_exp
            result['pts_pred_vis']   = pts_pred
            
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            # cycle loss (in the joint canonical space)
            #if opts.dist_corresp:
            result['frame_cyc_dis'] = (frame_cyc_dis * weights_coarse.detach()).sum(-1)
            #else:
            #    pts_exp_reg = pts_exp[:,None].detach()
            #    skin_forward = gauss_mlp_skinning(pts_exp_reg, embedding_xyz, 
            #                bones_rst,rest_pose_code,  nerf_skin, skin_aux=skin_aux)
            #    pts_exp_fw,_ = lbs(bones_rst, bone_rts_fw,
            #                      skin_forward, pts_exp_reg, backward=False)
            #    skin_backward = gauss_mlp_skinning(pts_exp_fw, embedding_xyz, 
            #                bones_dfm, time_embedded,  nerf_skin, skin_aux=skin_aux)
            #    pts_exp_fwbw,_ = lbs(bones_rst, bone_rts_fw,
            #                       skin_backward,pts_exp_fw)
            #    frame_cyc_dis = (pts_exp_fwbw - pts_exp_reg).norm(2,-1)
            #    result['frame_cyc_dis'] = sil_coarse.detach() * frame_cyc_dis[...,-1]
            if 'flowbw' in models.keys():
                result['frame_rigloss'] =  (frame_disp3d  * weights_coarse.detach()).sum(-1)
                # only evaluate at with_grad mode
                if xyz_coarse_frame.requires_grad:
                    # elastic energy
                    result['elastic_loss'] = elastic_loss(model_flowbw, embedding_xyz, 
                                      xyz_coarse_frame, time_embedded)
            else:
                result['frame_rigloss'] =  (frame_rigloss).mean(-1)

            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('compute cycle loss : %.2f'%(time.time()-start_time))     # 0.15 - 0.15 = 0.00
            
            ### script to plot sigmas/weights
            #from matplotlib import pyplot as plt
            #plt.ioff()
            #plt.plot(weights_coarse[weights_coarse.sum(-1)==1][:].T.cpu().numpy(),'*-')
            #plt.savefig('weights.png')
            #plt.cla()
            #plt.plot(sigmas[weights_coarse.sum(-1)==1][:].T.cpu().numpy(),'*-')
            #plt.savefig('sigmas.png')

        if is_training and 'nerf_vis' in models.keys():
            result['vis_loss'] = visibility_loss(models['nerf_vis'], embedding_xyz,
                            xyz_coarse_sampled, vis_coarse, obj_bound, chunk)

            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('compute visibility loss : %.2f'%(time.time()-start_time))        # 0.18 - 0.15 = 0.03

        # render flow 
        if 'rtk_vec_target' in rays.keys():
            if opts.dist_corresp:
                flo_coarse, flo_valid = vrender_flo(weights_coarse, xyz_coarse_target,
                                                    xys, img_size)
            else:
                flo_coarse = diff_flo(pts_target, xys, img_size)
                flo_valid = torch.ones_like(flo_coarse[...,:1])

            result['flo_coarse'] = flo_coarse
            result['flo_valid'] = flo_valid

            if opts.debug_detailed:
                torch.cuda.synchronize()
                print('render flow : %.2f'%(time.time()-start_time))                    # 0.18 - 0.18 = 0.00

        if 'rtk_vec_dentrg' in rays.keys():
            if opts.dist_corresp:
                fdp_coarse, fdp_valid = vrender_flo(weights_coarse, 
                                                    xyz_coarse_dentrg, xys, img_size)
            else:
                fdp_coarse = diff_flo(pts_dentrg, xys, img_size)
                fdp_valid = torch.ones_like(fdp_coarse[...,:1])
            result['fdp_coarse'] = fdp_coarse
            result['fdp_valid'] = fdp_valid

        if 'nerf_unc' in models.keys():
            # xys: bs,nsample,2
            # t: bs
            nerf_unc = models['nerf_unc']
            ts = rays['ts']
            vid_code = rays['vid_code']

            # change according to K
            xysn = rays['xysn']
            xyt = torch.cat([xysn, ts],-1)
            xyt_embedded = embedding_xyz(xyt)
            xyt_code = torch.cat([xyt_embedded, vid_code],-1)
            unc_pred = nerf_unc(xyt_code)
            #TODO add activation function
            #unc_pred = F.softplus(unc_pred)
            result['unc_pred'] = unc_pred
        
        if 'img_at_samp' in rays.keys():
            # compute other losses
            img_at_samp = rays['img_at_samp']
            sil_at_samp = rays['sil_at_samp']
            vis_at_samp = rays['vis_at_samp']
            flo_at_samp = rays['flo_at_samp']
            cfd_at_samp = rays['cfd_at_samp']
            #############################################################
            ################ modified by Chonghyuk Song #################
            dep_at_samp = rays['dep_at_samp']
            conf_at_samp = rays['conf_at_samp']

            # depth loss (depth_rnd.shape = [N], dep_at_samp.shape = [N, 1], conf_at_samp = [N, 1])
            # TODO: automate the depth scaling factor
            #dep_scale = 0.2
            dep_loss_samp = (depth_rnd[..., None] - opts.dep_scale * dep_at_samp).pow(2) * (conf_at_samp >= 1.5) * (dep_at_samp < 4.0)       # depth confidence values: 0 = low, 1 = medium, 2 = high
            #############################################################
            #############################################################

            # img loss
            img_loss_samp = (rgb_coarse - img_at_samp).pow(2)
            
            # sil loss, weight sil loss based on # points
            if is_training and sil_at_samp.sum()>0 and (1-sil_at_samp).sum()>0:
                pos_wt = vis_at_samp.sum()/   sil_at_samp[vis_at_samp>0].sum()
                neg_wt = vis_at_samp.sum()/(1-sil_at_samp[vis_at_samp>0]).sum()
                sil_balance_wt = 0.5*pos_wt*sil_at_samp + 0.5*neg_wt*(1-sil_at_samp)
            else: sil_balance_wt = 1
            sil_loss_samp = (sil_coarse[...,None] - sil_at_samp).pow(2) * sil_balance_wt
            sil_loss_samp = sil_loss_samp * vis_at_samp
               
            # flo loss, confidence weighting: 30x normalized distance - 0.1x pixel error
            flo_loss_samp = (flo_coarse - flo_at_samp).pow(2).sum(-1)
            # hard-threshold cycle error
            sil_at_samp_flo = (sil_at_samp>0)\
                     & (flo_valid==1)
            sil_at_samp_flo[cfd_at_samp==0] = False 
            if sil_at_samp_flo.sum()>0:
                cfd_at_samp = cfd_at_samp / cfd_at_samp[sil_at_samp_flo].mean()
            flo_loss_samp = flo_loss_samp[...,None] * cfd_at_samp
       
            result['img_at_samp']   = img_at_samp
            result['sil_at_samp']   = sil_at_samp
            result['vis_at_samp']   = vis_at_samp
            result['sil_at_samp_flo']   = sil_at_samp_flo
            result['flo_at_samp']   = flo_at_samp
            result['img_loss_samp'] = img_loss_samp 
            result['sil_loss_samp'] = sil_loss_samp
            result['flo_loss_samp'] = flo_loss_samp
            #############################################################
            ################ modified by Chonghyuk Song #################
            result['dep_loss_samp'] = dep_loss_samp
            
            '''
            with torch.no_grad():
                print("average rendered depth: {}".format(depth_rnd.mean()))
                print("average gt depth: {}".format(dep_at_samp.mean()))
                print("depth_rnd.shape: {}".format(depth_rnd.shape))
                print("dep_at_samp.shape: {}".format(dep_at_samp.shape))
                print("conf_at_samp.shape: {}".format(conf_at_samp.shape))
                print("rgb_coarse.shape: {}".format(rgb_coarse.shape))
                print("sil_coarse.shape: {}".format(sil_coarse.shape))
                print("dep_loss_samp.shape: {}".format(dep_loss_samp.shape))
                print("img_loss_samp.shape: {}".format(img_loss_samp.shape))
                print("sil_loss_samp.shape: {}".format(sil_loss_samp.shape))
            '''
            #############################################################
            #############################################################

    return result, weights_coarse


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


