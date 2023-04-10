########################## Written by Chonghyuk Song (04/17/22) ###########################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
from collections import defaultdict
import os
import os.path as osp
import pickle
import sys
sys.path.insert(0, 'third_party')
import cv2, numpy as np, time, torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import trimesh, pytorch3d, pytorch3d.loss, pdb
from pytorch3d import transforms
import configparser
import json
import copy
import functools

from nnutils.banmo import banmo
from nnutils.nerf import Embedding, NeRF, RTHead, SE3head, RTExplicit, Encoder,\
                    ScoreHead, Transhead, NeRFUnc, \
                    grab_xyz_weights, FrameCode, RTExpMLP, \
                    multi_view_embed_net, MultiViewFrameCode
from nnutils.geom_utils import K2mat, mat2K, Kmatinv, K2inv, raycast, sample_xy,\
                                chunk_rays, generate_bones,\
                                canonical2ndc, obj_to_cam, vec_to_sim3, \
                                near_far_to_bound, compute_flow_geodist, \
                                compute_flow_cse, fb_flow_check, pinhole_cam, \
                                render_color, mask_aug, bbox_dp2rnd, resample_dp, \
                                vrender_flo, get_near_far, array2tensor, rot_angle, \
                                rtk_invert, rtk_compose, bone_transform, correct_bones,\
                                correct_rest_pose, fid_reindex
from nnutils.rendering import render_rays_objs
from nnutils.loss_utils import eikonal_loss, rtk_loss, dense_truncated_eikonal_loss, \
                            feat_match_loss, kp_reproj_loss, grad_update_bone, \
                            loss_filter, loss_filter_line, compute_xyz_wt_loss,\
                            compute_root_sm_2nd_loss, shape_init_loss
from utils.io import draw_pts

class scene(nn.Module):
    def __init__(self, opts_list, data_info):
        super(scene, self).__init__()

        '''
        # modified "opts" for background object
        self.opts_bkgd = copy.deepcopy(opts)
        self.opts_bkgd.feat_wt = 0
        self.opts_bkgd.proj_wt = 0
        self.opts_bkgd.sil_wt = 0
        
        self.opts_bkgd.recon_bkgd = True
        self.opts_bkgd.loss_flt = False         # no loss filtering
        self.opts_bkgd.rm_novp = False          # no filtering with silhouette
        self.opts_bkgd.lbs = False              # no linear blend skinning (assume static bkgd)
        self.opts_bkgd.use_proj = False         # no use of re-projection loss
        self.opts_bkgd.use_unc = False          # no use of active sampling via uncertainty prediction net
        self.opts_bkgd.dist_corresp = False     
        self.opts_bkgd.unc_filter = False 
        '''      

        self.opts_list = opts_list

        # nn.ModuleList containing the banmo models for the foreground and background objects
        self.objs = nn.ModuleList()

        for obj_index, opts in enumerate(opts_list):
            self.objs.append(banmo(opts, data_info, obj_index))

        # any object agnostic settings (e.g. training settings) will be taken henceforth from the background opts (index = -1)
        self.loss_select = 1 # by default,  use all losses
        self.root_update = 1 # by default, update root pose
        self.body_update = 1 # by default, update body pose
        self.shape_update = 0 # by default, update all
        self.cvf_update = 0 # by default, update all
        self.progress = 0. # also reseted in optimizer
        self.counter_frz_rebone = 0. # counter to freeze params for reinit bones
        self.img_size = opts_list[-1].img_size  # current rendering size, 
                                                # have to be consistent with dataloader, 
                                                # eval/train has different size
        self.device = torch.device("cuda:%d"%opts_list[-1].local_rank)

    #################################################################
    ################## modified by Chonghyuk Song ###################
    def forward_warmup_shape_objs(self, batch):
        """
        batch variable is never used here
        """
        # render ground-truth data
        opts_list = self.opts_list
        
        # loss
        shape_factor = 0.1
        total_losses = []
        aux_out={}
        
        for obj_index, obj in enumerate(self.objs):
            opts_obj = opts_list[obj_index]

            if opts_obj.warmup_shape_ep > 0:
                
                if opts_obj.disentangled_nerf:
                    embed_obj = obj.embedding_xyz_sigmargb
                else:
                    embed_obj = obj.embedding_xyz

                loss_obj = shape_init_loss(obj.dp_verts_unit*shape_factor,obj.dp_faces, \
                                    obj.nerf_coarse, embed_obj,
                    bound_factor=opts_obj.bound_factor * 1.2, use_ellips=opts_obj.init_ellips)
                
                aux_out['shape_init_loss_obj{}'.format(obj_index)] = loss_obj
                total_losses.append(loss_obj)
        #############################################################
        #############################################################
        total_loss = torch.sum(torch.stack(total_losses))

        return total_loss, aux_out
    
    def forward_default(self, batch):
        opts_list = self.opts_list
        opts = opts_list[-1]

        # get root poses
        if all([opts_obj.samertktraj_opt for opts_obj in opts_list]):
            # all objects optimize the same, root-body poses of the bkgd
            rtk_all_objs = [self.objs[-1].compute_rts() for _ in range(len(self.objs))]
        else:
            # each object optimizes its own trajectory of poses
            rtk_all_objs = [obj.compute_rts() for obj in self.objs]

        # change near-far plane for all views for all objects
        if self.progress>=opts.nf_reset:
            rtk_np_objs = [rtk_all.clone().detach().cpu().numpy() for rtk_all in rtk_all_objs]
            valid_rts_objs = [obj.latest_vars['idk'].astype(bool) for obj in self.objs]

            for obj_index, obj in enumerate(self.objs):
                valid_rts = valid_rts_objs[obj_index]
                rtk_np = rtk_np_objs[obj_index]

                obj.latest_vars['rtk'][valid_rts, :3] = rtk_np[valid_rts]
                obj.near_far.data = get_near_far(obj.near_far.data, obj.latest_vars)

        if opts.debug:
            torch.cuda.synchronize()
            start_time = time.time()
        
        if opts.lineload:
            load_line = True
        else:
            load_line = False
        
        for obj_index, obj in enumerate(self.objs):

            # "set_input" does the following:
            # a) sets the contents of "batch" as attributes of object
            # b) scales the camera poses (rtk) stored as attributes in the object and \
            #    then computes the delta pose by calling its own nerf_root_rts
            # c) computes rtk, Kaug and save rtk, rt_raw into object.latest_vars
            bs = obj.set_input(obj_index, batch, load_line = load_line)

        if opts.debug:
            torch.cuda.synchronize()
            print('set input time:%.2f'%(time.time()-start_time))

        embedid = self.objs[-1].embedid
        
        if all([opts_obj.samertktraj_opt for opts_obj in opts_list]):
            # all objects optimize the same, root-body poses of the bkgd
            rtk_objs = [self.objs[-1].rtk for _ in range(len(self.objs))]
            kaug_objs = [self.objs[-1].kaug for _ in range(len(self.objs))]
        else:
            # each object optimizes its own trajectory of poses
            rtk_objs = [obj.rtk for obj in self.objs]
            kaug_objs = [obj.kaug for obj in self.objs]
        
        aux_out = {}

        # Render
        rendered, rand_inds = self.nerf_render(rtk_objs, kaug_objs, embedid, 
                nsample=opts.nsample, ndepth=opts.ndepth)
        
        if opts.debug:
            torch.cuda.synchronize()
            print('set input + render time:%.2f'%(time.time()-start_time))
        
        # image loss (we're not going to use the silhouette loss for fg-bkgd reconstruction)
        #sil_at_samp = rendered['sil_at_samp']
        sil_at_samp_flo = rendered['sil_at_samp_flo']
        #vis_at_samp = rendered['vis_at_samp']      # for the multi-object version of this codebase, we're using ['vis_at_samp'] of the bkgd object
        
        # frame-level rejection of bad segmentations for single-object version
        '''
        if opts.loss_flt:
            # frame-level rejection of bad segmentations
            if opts.lineload:
                invalid_idx = loss_filter_line(self.latest_vars['sil_err'],
                                               self.errid.long(),self.frameid.long(),
                                               rendered['sil_loss_samp']*opts.sil_wt,
                                               opts.img_size)
            else:
                sil_err, invalid_idx = loss_filter(self.latest_vars['sil_err'], 
                                             rendered['sil_loss_samp']*opts.sil_wt,
                                             sil_at_samp>-1, scale_factor=10)
                self.latest_vars['sil_err'][self.errid.long()] = sil_err
            
            if self.progress > (opts.warmup_steps):
                rendered['sil_loss_samp'][invalid_idx] *= 0.
                if invalid_idx.sum()>0:
                    print('%d removed from sil'%(invalid_idx.sum()))
        '''

        invalid_idx_objs = {}                                                   #manual update (04/11 commit from banmo repo)
        # frame-level rejection of bad segmentations for multi-object version
        for obj_index, opts_obj in enumerate(self.opts_list):
            sil_at_samp = rendered['sil_at_samp_obj{}'.format(obj_index)]
            if opts_obj.loss_flt:
                # frame-level rejection of bad segmentations
                if opts_obj.lineload:
                    invalid_idx = loss_filter_line(self.objs[obj_index].latest_vars['sil_err'],
                                                self.errid.long(), self.frameid.long(),
                                                rendered['sil_loss_samp_obj{}'.format(obj_index)]*opts_obj.sil_wt,
                                                opts.img_size)
                else:
                    sil_err, invalid_idx = loss_filter(self.objs[obj_index].latest_vars['sil_err'], 
                                                rendered['sil_loss_samp_obj{}'.format(obj_index)]*opts_obj.sil_wt,
                                                sil_at_samp>-1, scale_factor=10)
                    self.objs[obj_index].latest_vars['sil_err'][self.errid.long()] = sil_err
                
                if self.progress > (opts_obj.warmup_steps):
                    rendered['sil_loss_samp_obj{}'.format(obj_index)][invalid_idx] *= 0.
                    if invalid_idx.sum()>0:
                        print('%d removed from sil'%(invalid_idx.sum()))
                
                invalid_idx_objs[obj_index] = invalid_idx                       #manual update (04/11 commit from banmo repo)

        img_loss_samp = opts.img_wt*rendered['img_loss_samp']
        img_loss = img_loss_samp
        
        # rejection of frames with bad segmentations (frame-level rejection)    #manual update (04/11 commit from banmo repo)
        for obj_index, opts_obj in enumerate(self.opts_list):                   #manual update (04/11 commit from banmo repo)
            if opts_obj.loss_flt:                                               #manual update (04/11 commit from banmo repo)
                img_loss_samp[invalid_idx[obj_index]] *= 0                      #manual update (04/11 commit from banmo repo)

        # we're not gonna use this for multi-object version of the code base
        if opts.rm_novp:
            img_loss = img_loss * rendered['sil_coarse'].detach()       # won't have much of an effect as rendered['sil_coarse'] will almost always be 1's everywhere
        
        if len(self.opts_list) > 1:
            #sil_loss_samp = [rendered['sil_loss_samp_obj{}'.format(obj_index)][rendered['vis_at_samp_obj{}'.format(obj_index)]>0] for obj_index in range(len(opts_list)-1)]                                    # let's exclude the sil loss for the bkgd for now
            sil_loss_samp = [opts_obj.sil_wt * rendered['sil_loss_samp_obj{}'.format(obj_index)][rendered['vis_at_samp_obj{}'.format(obj_index)]>0] for obj_index, opts_obj in enumerate(opts_list[:-1])]       # let's exclude the sil loss for the bkgd for now
        else:
            #sil_loss_samp = [rendered['sil_loss_samp_obj{}'.format(obj_index)][rendered['vis_at_samp_obj{}'.format(obj_index)]>0] for obj_index in range(len(opts_list))]                                      # this is when we render a single object
            sil_loss_samp = [opts_obj.sil_wt * rendered['sil_loss_samp_obj{}'.format(obj_index)][rendered['vis_at_samp_obj{}'.format(obj_index)]>0] for obj_index, opts_obj in enumerate(opts_list)]            # this is when we render a single object
            
        sil_loss_samp = torch.sum(torch.stack(sil_loss_samp, axis = 0), axis = 0)
        sil_loss = sil_loss_samp.mean()
        
        # single-obj case                    (pre 10/09/22)   
        #if opts.sil_filter:                                                                    #manual update (04/11 commit from banmo repo)
        #    sil_at_samp = rendered['sil_at_samp_obj0']                                         #manual update (04/11 commit from banmo repo)
        #    img_loss = img_loss[sil_at_samp[...,0]>0].mean()    # eval on valid pts            #manual update (04/11 commit from banmo repo)
        #else:                                                                                  #manual update (04/11 commit from banmo repo)
        #    img_loss = img_loss.mean()                          # eval on all pixels           #manual update (04/11 commit from banmo repo)
        #manual update (04/11 commit from banmo repo)                                               
        #sil_at_samp_union_allobjs = functools.reduce(torch.logical_or, [rendered['sil_at_samp_obj{}'.format(obj_index)] for obj_index in range(len(opts_list))])
        
        #multi-obj case                    (post 10/09/22)
        sil_at_samp_union_allobjs = functools.reduce(torch.logical_or, [(rendered['sil_at_samp_obj{}'.format(obj_index)] > 0) for obj_index in range(len(opts_list))])    
        rendered['sil_at_samp_union_allobjs'] = sil_at_samp_union_allobjs

        img_loss = img_loss[sil_at_samp_union_allobjs[..., 0]> 0].mean()       #manual update (04/11 commit from banmo repo)    
        
        #sil_loss_samp = opts.sil_wt*rendered['sil_loss_samp']
        #sil_loss = sil_loss_samp[vis_at_samp>0].mean()

        aux_out['sil_loss'] = sil_loss
        aux_out['img_loss'] = img_loss
        total_loss = img_loss
        total_loss = total_loss + sil_loss

        # depth reconstruction loss
        dep_loss_samp = rendered['dep_loss_samp']
        dep_loss = dep_loss_samp

        # rejection of frames with bad segmentations (frame-level rejection)    #manual update (04/11 commit from banmo repo)
        for obj_index, opts_obj in enumerate(self.opts_list):                   #manual update (04/11 commit from banmo repo)
            if opts_obj.loss_flt:                                               #manual update (04/11 commit from banmo repo)
                dep_loss_samp[invalid_idx[obj_index]] *= 0                      #manual update (04/11 commit from banmo repo)

        if opts.rm_novp:
            dep_loss = dep_loss * rendered['sil_coarse'].detach()               # won't have much of an effect as rendered['sil_coarse'] will almost always be 1's everywhere

        #if opts.sil_filter:                                                                #manual update (04/11 commit from banmo repo)
        #    sil_at_samp = rendered['sil_at_samp_obj0']                                     #manual update (04/11 commit from banmo repo)
        #    dep_loss = dep_loss[sil_at_samp[...,0]>0].mean()    # eval on valid pts        #manual update (04/11 commit from banmo repo)   
        #else:                                                                              #manual update (04/11 commit from banmo repo)
        #    dep_loss = dep_loss.mean()                          # eval on all pts          #manual update (04/11 commit from banmo repo)
        dep_loss = dep_loss[sil_at_samp_union_allobjs[..., 0]> 0].mean()                    #manual update (04/11 commit from banmo repo)        

        aux_out['dep_loss_wo_weight'] = dep_loss            # so that we can see if the depth loss decreases even without any depth supervision
        dep_loss = opts.dep_wt * dep_loss                   
        aux_out['dep_loss'] = dep_loss
        
        total_loss = total_loss + dep_loss

        # flow loss
        if opts.use_corresp:
            #manual update (04/11 commit from banmo repo)
            # TODO: implement frame-level rejection of bad flows
            '''
            if opts.loss_flt:
                # find flow window
                dframe = (self.frameid.view(2,-1).flip(0).reshape(-1) - \
                          self.frameid).abs()                                                                   # self.frameid set right after self.sample_pxs_obj
                didxs = dframe.log2().long()
                for didx in range(6):
                    subidx = didxs==didx

                    #flo_err, invalid_idx = loss_filter(self.latest_vars['flo_err'][:,didx], 
                    #                                rendered['flo_loss_samp'][subidx],
                    #                                sil_at_samp_flo[subidx], scale_factor=20)
                    # TODO: generalize this for multiple objects where we use different Kaugs for each object
                    flo_err, invalid_idx = loss_filter(self.objs[-1].latest_vars['flo_err'][:,didx],            # for now let's just use the latest_vars['flo_err'] of background since we're assuming save Kaug for fg and bkgd
                                                    rendered['flo_loss_samp'][subidx],
                                                    sil_at_samp_flo[subidx], scale_factor=20)
                    self.objs[-1].latest_vars['flo_err'][self.errid.long()[subidx],didx] = flo_err
                    if self.progress > (opts.warmup_steps):
                        #print('%d removed from flow'%(invalid_idx.sum()))
                        flo_loss_samp_sub = rendered['flo_loss_samp'][subidx]
                        flo_loss_samp_sub[invalid_idx] *= 0.
                        rendered['flo_loss_samp'][subidx] = flo_loss_samp_sub
            '''
            flo_loss_samp = rendered['flo_loss_samp']

            for obj_index, opts_obj in enumerate(self.opts_list):                   #manual update (04/11 commit from banmo repo)
                if opts_obj.loss_flt:                                               #manual update (04/11 commit from banmo repo)
                    flo_loss_samp[invalid_idx[obj_index]] *= 0                      #manual update (04/11 commit from banmo repo)

            if opts.rm_novp:
                flo_loss_samp = flo_loss_samp * rendered['sil_coarse'].detach()

            flo_loss = flo_loss_samp[sil_at_samp_flo[...,0] & (sil_at_samp_union_allobjs[..., 0]> 0)].mean() * 2             # eval on pts with valid flow and pts inside union of mask across all objects #manual update (04/11 commit from banmo repo)
            flo_loss = flo_loss * opts.flow_wt

            # warm up by only using flow loss to optimize root pose
            if self.loss_select == 0:
                total_loss = total_loss*0. + flo_loss
            else:
                total_loss = total_loss + flo_loss
            aux_out['flo_loss'] = flo_loss

        # viser loss
        
        '''
        if opts.use_embed:
            feat_err_samp = rendered['feat_err']* opts.feat_wt
            if opts.loss_flt:
                if opts.lineload:
                    #invalid_idx = loss_filter_line(self.latest_vars['fp_err'][:,0],
                    #                           self.errid.long(),self.frameid.long(),
                    #                           feat_err_samp * sil_at_samp,
                    #                           opts.img_size, scale_factor=10)

                    # we could take the latest_vars['fp_err'] from any object we wanted
                    invalid_idx = loss_filter_line(self.objs[0].latest_vars['fp_err'][:,0],
                                               self.objs[0].errid.long(),self.objs[0].frameid.long(),
                                               feat_err_samp * sil_at_samp,
                                               opts.img_size, scale_factor=10)
                else:
                    # loss filter
                    #feat_err, invalid_idx = loss_filter(self.latest_vars['fp_err'][:,0], 
                    #                                feat_err_samp,
                    #                                sil_at_samp>0)
                    feat_err, invalid_idx = loss_filter(self.objs[0].latest_vars['fp_err'][:,0], 
                                                    feat_err_samp,
                                                    sil_at_samp>0)

                    #self.latest_vars['fp_err'][self.errid.long(),0] = feat_err
                    for obj in self.objs:
                        obj.latest_vars['fp_err'][obj.errid.long(),0] = feat_err

                if self.progress > (opts.warmup_steps):
                    feat_err_samp[invalid_idx] *= 0.
                    if invalid_idx.sum()>0:
                        print('%d removed from feat'%(invalid_idx.sum()))
            
            feat_loss = feat_err_samp
            feat_loss = feat_loss[sil_at_samp>0].mean()
            total_loss = total_loss + feat_loss
            aux_out['feat_loss'] = feat_loss

            aux_out['beta_feat'] = self.nerf_feat.beta.clone().detach()[0]
        '''

        for obj_index, opts_obj in enumerate(self.opts_list):
            sil_at_samp = rendered['sil_at_samp_obj{}'.format(obj_index)]
            
            if opts_obj.use_embed:
                # feature matching loss, feature rendering loss 
                #feat_err_samp = rendered['feat_err_obj{}'.format(obj_index)]* opts_obj.feat_wt
                frnd_loss_samp = rendered['frnd_loss_samp_obj{}'.format(obj_index)]* opts_obj.frnd_wt                   #manual update (04/11 commit from banmo repo)

                if opts_obj.loss_flt:
                    #manual update (04/11 commit from banmo repo)
                    '''
                    if opts_obj.lineload:
                        #invalid_idx = loss_filter_line(self.latest_vars['fp_err'][:,0],
                        #                           self.errid.long(),self.frameid.long(),
                        #                           feat_err_samp * sil_at_samp,
                        #                           opts.img_size, scale_factor=10)

                        # we could take the latest_vars['fp_err'] from any object we wanted
                        invalid_idx = loss_filter_line(self.objs[obj_index].latest_vars['fp_err'][:,0],
                                                    self.errid.long(), self.frameid.long(),
                                                    feat_err_samp * sil_at_samp,
                                                    opts.img_size, scale_factor=10)
                    else:
                        # loss filter
                        #feat_err, invalid_idx = loss_filter(self.latest_vars['fp_err'][:,0], 
                        #                                feat_err_samp,
                        #                                sil_at_samp>0)
                        feat_err, invalid_idx = loss_filter(self.objs[obj_index].latest_vars['fp_err'][:,0], 
                                                        feat_err_samp,
                                                        sil_at_samp>0)

                        #self.latest_vars['fp_err'][self.errid.long(),0] = feat_err
                        self.objs[obj_index].latest_vars['fp_err'][obj.errid.long(),0] = feat_err

                    if self.progress > (opts.warmup_steps):
                        feat_err_samp[invalid_idx] *= 0.
                        if invalid_idx.sum()>0:
                            print('%d removed from feat'%(invalid_idx.sum()))
                    '''
                    #feat_err_samp[invalid_idx_objs[obj_index]] *= 0                     #manual update (04/11 commit from banmo repo)
                    frnd_loss_samp[invalid_idx_objs[obj_index]] *= 0                    #manual update (04/11 commit from banmo repo)

                #feat_loss = feat_err_samp
                if opts.rm_novp:
                    #feat_loss = feat_loss * rendered['sil_coarse_obj{}'.format(obj_index)].detach()
                    frnd_loss_samp = frnd_loss_samp * rendered['sil_coarse_obj{}'.format(obj_index)].detach()           #manual update (04/11 commit from banmo repo)

                #feat_loss = feat_loss[sil_at_samp>0].mean()
                # single fg-obj case (pre 10/09/22)
                # feat_rnd_loss = frnd_loss_samp[sil_at_samp[...,0]>0].mean()             #manual update (04/11 commit from banmo repo)
                
                # multi fg-obj case (post 10/09/22)
                feat_rnd_loss = frnd_loss_samp[(0 < sil_at_samp[...,0]) & (sil_at_samp[...,0] < 254)]                   # this implies that for fg reconstruction, we're ignoring losses for frames with invalid masks -> this will be compensated for during joint-finetuning by training on 2d features
                feat_rnd_loss = 0 if feat_rnd_loss.nelement == 0 else feat_rnd_loss.mean()     # eval on valid pts (if there are no valid pts, return 0)

                #total_loss = total_loss + feat_loss
                total_loss = total_loss + feat_rnd_loss                                 #manual update (04/11 commit from banmo repo)

                #aux_out['feat_loss{}'.format(obj_index)] = feat_loss
                aux_out['feat_rnd_loss{}'.format(obj_index)] = feat_rnd_loss            #manual update (04/11 commit from banmo repo)
                aux_out['beta_feat{}'.format(obj_index)] = self.objs[obj_index].nerf_feat.beta.clone().detach()[0]

            if opts_obj.use_proj:
                proj_err_samp = rendered['proj_err_obj{}'.format(obj_index)]* opts_obj.proj_wt

                if opts_obj.loss_flt:
                    #manual update (04/11 commit from banmo repo)
                    '''
                    if opts.lineload:
                        #invalid_idx = loss_filter_line(self.latest_vars['fp_err'][:,1],
                        #                           self.errid.long(),self.frameid.long(),
                        #                           proj_err_samp * sil_at_samp,
                        #                           opts.img_size, scale_factor=10)
                        invalid_idx = loss_filter_line(self.objs[obj_index].latest_vars['fp_err'][:,1],
                                                self.errid.long(),self.frameid.long(),
                                                proj_err_samp * sil_at_samp,
                                                opts.img_size, scale_factor=10)
                        
                    else:
                        #proj_err, invalid_idx = loss_filter(self.latest_vars['fp_err'][:,1], 
                        #                                proj_err_samp,
                        #                                sil_at_samp>0)
                        proj_err, invalid_idx = loss_filter(self.objs[obj_index].latest_vars['fp_err'][:,1], 
                                                        proj_err_samp,
                                                        sil_at_samp>0)
                        
                        #self.latest_vars['fp_err'][self.errid.long(),1] = proj_err
                        self.objs[obj_index].latest_vars['fp_err'][obj.errid.long(),1] = proj_err                                                    

                    if self.progress > (opts.warmup_steps):
                        proj_err_samp[invalid_idx] *= 0.
                        if invalid_idx.sum()>0:
                            print('%d removed from proj'%(invalid_idx.sum()))
                    '''
                    proj_err_samp[invalid_idx] *= 0.                            #manual update (04/11 commit from banmo repo)

                # single fg-obj case (pre 10/09/22)
                #proj_loss = proj_err_samp[sil_at_samp>0].mean()

                # multi fg-obj case (post 10/09/22)
                proj_loss = proj_err_samp[(0 < sil_at_samp[...,0]) & (sil_at_samp[...,0] < 254)]                 # this implies that for fg reconstruction, we're ignoring losses for frames with invalid masks -> this will be compensated for during joint-finetuning by training on 2d features
                proj_loss = 0 if proj_loss.nelement == 0 else proj_loss.mean()      # eval on valid pts (if there are no valid pts, return 0)

                aux_out['proj_loss_obj{}'.format(obj_index)] = proj_loss
                
                if opts_obj.freeze_proj:
                    total_loss = total_loss + proj_loss
                    ## warm up by only using projection loss to optimize bones
                    warmup_weight = (self.progress - opts_obj.proj_start)/(opts_obj.proj_end-opts_obj.proj_start)
                    warmup_weight = (warmup_weight - 0.8) * 5 #  [-4,1]
                    warmup_weight = np.clip(warmup_weight, 0,1)
                    if (self.progress > opts_obj.proj_start and \
                        self.progress < opts_obj.proj_end):
                        total_loss = total_loss*warmup_weight + \
                                10*proj_loss*(1-warmup_weight)
                else:
                    # only add it after feature volume is trained well
                    total_loss = total_loss + proj_loss
            
            # assume we’ll set use_unc flag on an obj-by-obj basis, as opposed to setting it unilaterally for all objects (e.g. we’ll disable use_unc for the bkgd, and focus on finetuning the foreground)
            if opts_obj.use_unc:
                if 'unc_pred_obj{}'.format(obj_index) in rendered.keys():
                    unc_pred = rendered['unc_pred_obj{}'.format(obj_index)]
                    
                    # single-fg-obj case
                    #unc_rgb = sil_at_samp[...,0]*img_loss_samp.mean(-1)

                    # multi-fg-obj case
                    unc_rgb = ((0 < sil_at_samp[...,0]) & (sil_at_samp[...,0] < 254)) * img_loss_samp.mean(-1)
                    unc_dep = ((0 < sil_at_samp[...,0]) & (sil_at_samp[...,0] < 254)) * dep_loss_samp.mean(-1)          # remember, dep_loss_samp, already takes into account conf_at_samp

                    #manual update (04/11 commit from banmo repo)
                    '''
                    if opts_obj.use_embed:
                        unc_feat= (sil_at_samp*feat_err_samp)[...,0]
                        unc_accumulated = unc_accumulated + unc_feat

                    if opts_obj.use_proj:
                        unc_proj= (sil_at_samp*proj_err_samp)[...,0]
                        unc_accumulated = unc_accumulated + unc_proj
                    '''

                    if opts_obj.use_unc_depth:
                        unc_accumulated = unc_dep
                    else:
                        unc_accumulated = unc_rgb

                    unc_loss = (unc_accumulated.detach() - unc_pred[...,0]).pow(2)
                    unc_loss = unc_loss.mean()
                    aux_out['unc_loss_obj{}'.format(obj_index)] = unc_loss
                    total_loss = total_loss + unc_loss

        # use_proj for foreground only (i.e. use silhouette filtering with sil_at_samp)
        '''
        if opts.use_proj:
            proj_err_samp = rendered['proj_err']* opts.proj_wt
            if opts.loss_flt:
                if opts.lineload:
                    #invalid_idx = loss_filter_line(self.latest_vars['fp_err'][:,1],
                    #                           self.errid.long(),self.frameid.long(),
                    #                           proj_err_samp * sil_at_samp,
                    #                           opts.img_size, scale_factor=10)
                    invalid_idx = loss_filter_line(self.objs[0].latest_vars['fp_err'][:,1],
                                               self.objs[0].errid.long(),self.objs[0].frameid.long(),
                                               proj_err_samp * sil_at_samp,
                                               opts.img_size, scale_factor=10)
                    
                else:
                    #proj_err, invalid_idx = loss_filter(self.latest_vars['fp_err'][:,1], 
                    #                                proj_err_samp,
                    #                                sil_at_samp>0)
                    proj_err, invalid_idx = loss_filter(self.objs[0].latest_vars['fp_err'][:,1], 
                                                    proj_err_samp,
                                                    sil_at_samp>0)
                    
                    #self.latest_vars['fp_err'][self.errid.long(),1] = proj_err
                    for i, obj in enumerate(self.objs):
                        # the background object doesn't have any bones
                        if i == len(self.objs) - 1:
                            break
                        obj.latest_vars['fp_err'][obj.errid.long(),1] = proj_err                                                    

                if self.progress > (opts.warmup_steps):
                    proj_err_samp[invalid_idx] *= 0.
                    if invalid_idx.sum()>0:
                        print('%d removed from proj'%(invalid_idx.sum()))

            proj_loss = proj_err_samp[sil_at_samp>0].mean()
            aux_out['proj_loss'] = proj_loss
            if opts.freeze_proj:
                total_loss = total_loss + proj_loss
                ## warm up by only using projection loss to optimize bones
                warmup_weight = (self.progress - opts.proj_start)/(opts.proj_end-opts.proj_start)
                warmup_weight = (warmup_weight - 0.8) * 5 #  [-4,1]
                warmup_weight = np.clip(warmup_weight, 0,1)
                if (self.progress > opts.proj_start and \
                    self.progress < opts.proj_end):
                    total_loss = total_loss*warmup_weight + \
                               10*proj_loss*(1-warmup_weight)
            else:
                # only add it after feature volume is trained well
                total_loss = total_loss + proj_loss
        
        if 'frame_cyc_dis' in rendered.keys():
            # cycle loss
            cyc_loss = rendered['frame_cyc_dis'].mean()
            total_loss = total_loss + cyc_loss * opts.cyc_wt
            #total_loss = total_loss + cyc_loss*0
            aux_out['cyc_loss'] = cyc_loss

            # globally rigid prior
            rig_loss = 0.0001*rendered['frame_rigloss'].mean()
            if opts.rig_loss:
                total_loss = total_loss + rig_loss
            else:
                total_loss = total_loss + rig_loss*0
            aux_out['rig_loss'] = rig_loss

            # elastic energy for se3 field / translation field
            if 'elastic_loss' in rendered.keys():
                elastic_loss = rendered['elastic_loss'].mean() * 1e-3
                total_loss = total_loss + elastic_loss
                aux_out['elastic_loss'] = elastic_loss
        '''

        # TODO: we need silhouette filtering on cycle loss (NOT SURE IF ALREADY IMPLEMENTED IN inference_deform)
        # regularization
        for obj_index in range(len(self.opts_list)):
            if 'frame_cyc_dis_obj{}'.format(obj_index) in rendered.keys():
                # cycle loss
                cyc_loss = rendered['frame_cyc_dis_obj{}'.format(obj_index)].mean()
                total_loss = total_loss + cyc_loss * opts.cyc_wt
                #total_loss = total_loss + cyc_loss*0
                aux_out['cyc_loss_obj{}'.format(obj_index)] = cyc_loss

                # globally rigid prior
                if opts_list[obj_index].rig_loss:
                    rig_loss = 0.0001*rendered['frame_rigloss_obj{}'.format(obj_index)].mean()
                    total_loss = total_loss + rig_loss
                    aux_out['rig_loss_obj{}'.format(obj_index)] = rig_loss
                #else:
                #    total_loss = total_loss + rig_loss*0
                #aux_out['rig_loss_obj{}'.format(obj_index)] = rig_loss

                # elastic energy for se3 field / translation field
                if 'elastic_loss_obj{}'.format(obj_index) in rendered.keys():
                    elastic_loss = rendered['elastic_loss_obj{}'.format(obj_index)].mean() * 1e-3
                    total_loss = total_loss + elastic_loss
                    aux_out['elastic_loss_obj{}'.format(obj_index)] = elastic_loss

        # regularization of root poses
        if opts.root_sm:
            for obj_index, rtk_all in enumerate(rtk_all_objs):
                root_sm_loss = compute_root_sm_2nd_loss(rtk_all, self.objs[obj_index].data_offset)
                aux_out['root_sm_loss{}'.format(obj_index)] = root_sm_loss
                total_loss = total_loss + root_sm_loss

        # eikonal loss
        #if opts.eikonal_loss:
        for obj_index, obj in enumerate(self.objs):
            if opts_list[obj_index].eikonal_loss:
                #############################################################
                ################ modified by Chonghyuk Song #################
                if opts_list[obj_index].disentangled_nerf:
                    embed = obj.embedding_xyz_sigmargb
                else:
                    embed = obj.embedding_xyz
                #############################################################
                #############################################################

                # TODO: find a way to feed in the correct rendered['pts_exp_vis'] according to object
                #ekl_loss = 1e-5*eikonal_loss(obj.nerf_coarse, obj.embedding_xyz, 
                ekl_loss = 1e-5*eikonal_loss(obj.nerf_coarse, embed, 
                        rendered['pts_exp_vis_obj{}'.format(obj_index)], obj.latest_vars['obj_bound'])
                aux_out['ekl_loss{}'.format(obj_index)] = ekl_loss
                total_loss = total_loss + ekl_loss
        
            if opts_list[obj_index].dense_trunc_eikonal_loss:
                #############################################################
                ################ modified by Chonghyuk Song #################
                if opts_list[obj_index].disentangled_nerf:
                    embed = obj.embedding_xyz_sigmargb
                else:
                    embed = obj.embedding_xyz
                #############################################################
                #############################################################

                dense_trunc_ekl_loss = dense_truncated_eikonal_loss(obj.nerf_coarse, embed, opts_list[obj_index].ntrunc, rendered['xyz_trunc_region_obj{}'.format(obj_index)], rendered['conf_at_samp'])
                aux_out['dense_trunc_ekl_loss_wo_weight_obj{}'.format(obj_index)] = dense_trunc_ekl_loss
                dense_trunc_ekl_loss = opts_list[obj_index].dense_trunc_eikonal_wt * dense_trunc_ekl_loss
                aux_out['dense_trunc_ekl_loss_obj{}'.format(obj_index)] = dense_trunc_ekl_loss
                total_loss = total_loss + dense_trunc_ekl_loss

        # bone location regularization: pull bones away from empty space (low sdf)
        for obj_index, obj in enumerate(self.objs):

            if opts_list[obj_index].lbs and opts_list[obj_index].bone_loc_reg>0:                
                bones_rst = obj.bones
                bones_rst,_ = correct_bones(obj, bones_rst)
                mesh_rest = obj.latest_vars['mesh_rest']
                
                if len(mesh_rest.vertices)>100: # not a degenerate mesh
                    # issue #4 the following causes error on certain archs for torch110+cu113
                    # seems to be a conflict between geomloss and pytorch3d
                    # mesh_rest = pytorch3d.structures.meshes.Meshes(
                    #         verts=torch.Tensor(mesh_rest.vertices[None]),
                    #         faces=torch.Tensor(mesh_rest.faces[None]))
                    # a ugly workaround 
                    mesh_verts = [torch.Tensor(mesh_rest.vertices)]
                    mesh_faces = [torch.Tensor(mesh_rest.faces).long()]
                    try:
                        mesh_rest = pytorch3d.structures.meshes.Meshes(verts=mesh_verts, faces=mesh_faces)
                    except:
                        mesh_rest = pytorch3d.structures.meshes.Meshes(verts=mesh_verts, faces=mesh_faces)

                    shape_samp = pytorch3d.ops.sample_points_from_meshes(mesh_rest,
                                            1000, return_normals=False)
                    shape_samp = shape_samp[0].to(obj.device)
                    from geomloss import SamplesLoss
                    samploss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
                    bone_loc_loss = samploss(bones_rst[:,:3]*10, shape_samp*10)
                    bone_loc_loss = opts_list[obj_index].bone_loc_reg*bone_loc_loss
                    total_loss = total_loss + bone_loc_loss
                    aux_out['bone_loc_loss{}'.format(obj_index)] = bone_loc_loss
        
        # visibility loss
        # for now, we're just computing a visibility loss for the background, which has the greatest spatial extent out of all objects
        if 'vis_loss' in rendered.keys():
            vis_loss = 0.01*rendered['vis_loss'].mean()
            total_loss = total_loss + vis_loss
            aux_out['visibility_loss'] = vis_loss

        '''
        # uncertainty MLP inference (only apply it to foreground)
        if opts.use_unc:
            unc_pred = rendered['unc_pred']
            unc_rgb = sil_at_samp[...,0]*img_loss_samp.mean(-1)
            unc_feat= (sil_at_samp*feat_err_samp)[...,0]
            unc_proj= (sil_at_samp*proj_err_samp)[...,0]
            #unc_accumulated = unc_feat + unc_proj
            #unc_accumulated = unc_feat + unc_proj + unc_rgb*0.1
            unc_accumulated = unc_feat + unc_proj + unc_rgb
#            unc_accumulated = unc_rgb
#            unc_accumulated = unc_rgb + unc_sil

            unc_loss = (unc_accumulated.detach() - unc_pred[...,0]).pow(2)
            unc_loss = unc_loss.mean()
            aux_out['unc_loss'] = unc_loss
            total_loss = total_loss + unc_loss
        '''

        # cse feature tuning
        for obj_index, obj in enumerate(self.objs):
            if opts_list[obj_index].ft_cse and opts_list[obj_index].mt_cse:

                csenet_loss = (obj.csenet_feats - obj.csepre_feats).pow(2).sum(1)
                csenet_loss = csenet_loss[obj.dp_feats_mask].mean()* 1e-5
                if self.progress < opts_list[obj_index].mtcse_steps:
                    total_loss = total_loss*0 + csenet_loss
                else:
                    total_loss = total_loss + csenet_loss
                aux_out['csenet_loss{}'.format(obj_index)] = csenet_loss

        if opts.freeze_coarse:
            # foreground
            # compute nerf xyz wt loss

            for obj_index, obj in enumerate(self.objs):
                
                # foreground
                #if obj_index < len(self.objs) - 1:    
                if opts_list[obj_index].lbs:
                    # compute nerf xyz wt loss
                    shape_xyz_wt_curr = grab_xyz_weights(obj.nerf_coarse)
                    shape_xyz_wt_loss = 100*compute_xyz_wt_loss(obj.shape_xyz_wt, 
                                                                shape_xyz_wt_curr)
                    skin_xyz_wt_curr = grab_xyz_weights(obj.nerf_skin)
                    skin_xyz_wt_loss = 100*compute_xyz_wt_loss(obj.skin_xyz_wt, 
                                                                skin_xyz_wt_curr)
                    feat_xyz_wt_curr = grab_xyz_weights(obj.nerf_feat)
                    feat_xyz_wt_loss = 100*compute_xyz_wt_loss(obj.feat_xyz_wt, 
                                                                feat_xyz_wt_curr)
                    aux_out['shape_xyz_wt_loss{}'.format(obj_index)] = shape_xyz_wt_loss
                    aux_out['skin_xyz_wt_loss{}'.format(obj_index)] = skin_xyz_wt_loss
                    aux_out['feat_xyz_wt_loss{}'.format(obj_index)] = feat_xyz_wt_loss
                    total_loss = total_loss + shape_xyz_wt_loss + skin_xyz_wt_loss\
                            + feat_xyz_wt_loss

                # background
                else:
                    shape_xyz_wt_curr = grab_xyz_weights(obj.nerf_coarse)
                    shape_xyz_wt_loss = 100*compute_xyz_wt_loss(obj.shape_xyz_wt, 
                                                                shape_xyz_wt_curr)
                    aux_out['shape_xyz_wt_loss{}'.format(obj_index)] = shape_xyz_wt_loss
                    aux_out['skin_xyz_wt_loss{}'.format(obj_index)] = 0
                    aux_out['feat_xyz_wt_loss{}'.format(obj_index)] = 0

                    total_loss = total_loss + shape_xyz_wt_loss
        
        # add entropy loss
        if opts.use_ent:
            entropy_loss = rendered['entropy_loss_samp']
            aux_out['entropy_loss_wo_weight'] = entropy_loss
            total_loss = total_loss + entropy_loss * opts.ent_wt

        # save some variables
        for obj_index, obj in enumerate(self.objs):
            if opts_list[obj_index].lbs:
                #aux_out['skin_scale'] = self.skin_aux[0].clone().detach()
                #aux_out['skin_const'] = self.skin_aux[1].clone().detach()
                aux_out['skin_scale{}'.format(obj_index)] = obj.skin_aux[0].clone().detach()
                aux_out['skin_const{}'.format(obj_index)] = obj.skin_aux[1].clone().detach()

        total_loss = total_loss * opts.total_wt
        aux_out['total_loss'] = total_loss
    
        for obj_index, obj in enumerate(self.objs):            
            aux_out['beta{}'.format(obj_index)] = obj.nerf_coarse.beta.clone().detach()[0]
        
        if opts.debug:
            torch.cuda.synchronize()
            print('set input + render + loss time:%.2f'%(time.time()-start_time))

        return total_loss, aux_out
    #################################################################
    #################################################################

    def nerf_render(self, rtk_objs, kaug_objs, embedid, nsample=256, ndepth=128):
        # rtks: a list of rtks
        # kaugs: a list of kaugs
        # TODO: fix 
        opts = self.opts_list[-1]

        if opts.debug:
            torch.cuda.synchronize()
            start_time = time.time()

        # 2bs,...
        # Rmat, Tmat, Kinv = self.prepare_ray_cams(rtk, kaug)
        # bs = Kinv.shape[0]
        Rmat_objs = []
        Tmat_objs = []
        Kinv_objs = []

        for obj_index, (rtk, kaug) in enumerate(zip(rtk_objs, kaug_objs)):
            Rmat, Tmat, Kinv = self.prepare_ray_cams(rtk, kaug)   # since prepare_ray_cams is a static method doesn't matter from which object we call the method from

            Rmat_objs.append(Rmat)
            Tmat_objs.append(Tmat)
            Kinv_objs.append(Kinv)
        
        bs = Kinv.shape[0]

        # sample pxs (rays)
        # for batch:2bs,            nsample+x
        # for line: 2bs*(nsample+x),1
        # TODO: implement sample_pxs for foreground - background reconstruction
        #rand_inds, rays, frameid, errid = self.sample_pxs(bs, nsample, Rmat, Tmat, Kinv,
        #self.dataid, self.frameid, self.frameid_sub, self.embedid,self.lineid,self.errid,
        #self.imgs, self.masks, self.deps, self.confs, self.vis2d, self.flow, self.occ, self.dp_feats)
        rand_inds, rays_objs, frameid, errid = self.sample_pxs_objs(bs, nsample, Rmat_objs, Tmat_objs, Kinv_objs)
        self.frameid = frameid # only used in loss filter
        self.errid = errid

        if opts.debug:
            torch.cuda.synchronize()
            print('prepare rays time: %.2f'%(time.time()-start_time))

        #bs_rays = rays['bs'] * rays['nsample'] # over pixels
        bs_rays = rays_objs[-1]['bs'] * rays_objs[-1]['nsample'] # over pixels

        results=defaultdict(list)
        for i in range(0, bs_rays, opts.chunk):
            rays_chunk_objs = []

            # rays_chunk = chunk_rays(rays,i,opts.chunk)
            for rays in rays_objs:
                rays_chunk = chunk_rays(rays,i,opts.chunk)
                rays_chunk_objs.append(rays_chunk)

            # decide whether to use fine samples 
            #if self.progress > opts.fine_steps:
            if self.progress > opts.fine_steps:
                self.use_fine = True
            else:
                self.use_fine = False

            # TODO: modify to accomodate multiple models
            '''
            rendered_chunks = render_rays(self.nerf_models,
                        self.embeddings,
                        rays_chunk,
                        N_samples = ndepth,
                        use_disp=False,
                        perturb=opts.perturb,
                        noise_std=opts.noise_std,
                        chunk=opts.chunk, # chunk size is effective in val mode
                        obj_bound=self.latest_vars['obj_bound'],
                        use_fine=self.use_fine,
                        img_size=self.img_size,
                        progress=self.progress,
                        opts=opts,
                        )
            '''
            rendered_chunks = render_rays_objs([obj.nerf_models for obj in self.objs],
                        [obj.embeddings for obj in self.objs],
                        rays_chunk_objs,
                        N_samples = ndepth,
                        perturb=opts.perturb,
                        noise_std=opts.noise_std,
                        chunk=opts.chunk, # chunk size is effective in val mode
                        obj_bounds=[obj.latest_vars['obj_bound'] for obj in self.objs],
                        use_fine=self.use_fine,
                        img_size=self.img_size,
                        progress=self.progress,
                        opts=opts,
                        opts_objs=self.opts_list,
                        )
            for k, v in rendered_chunks.items():
                results[k] += [v]
        
        for k, v in results.items():
            if v[0].dim()==0: # loss
                v = torch.stack(v).mean()
            else:
                v = torch.cat(v, 0)
                if self.training:
                    v = v.view(rays['bs'],rays['nsample'],-1)
                else:
                    v = v.view(bs,self.img_size, self.img_size, -1)
            results[k] = v
        if opts.debug:
            torch.cuda.synchronize()
            print('rendering time: %.2f'%(time.time()-start_time))

        # viser feature matching: TODO: how to scale this to foreground background reconstruction
        # temporarily disabling this line of code just to do eval composite rendering
        '''
        if opts.use_embed:
            results['pts_pred'] = (results['pts_pred'] - torch.Tensor(self.vis_min[None]).\
                    to(self.device)) / torch.Tensor(self.vis_len[None]).to(self.device)
            results['pts_exp']  = (results['pts_exp'] - torch.Tensor(self.vis_min[None]).\
                    to(self.device)) / torch.Tensor(self.vis_len[None]).to(self.device)
            results['pts_pred'] = results['pts_pred'].clamp(0,1)
            results['pts_exp']  = results['pts_exp'].clamp(0,1)
        '''

        if opts.debug:
            torch.cuda.synchronize()
            print('compute flow time: %.2f'%(time.time()-start_time))
        return results, rand_inds

    @staticmethod
    def prepare_ray_cams(rtk, kaug):
        """ 
        in: rtk, kaug
        out: Rmat, Tmat, Kinv
        """
        Rmat = rtk[:,:3,:3]
        Tmat = rtk[:,:3,3]

        # hardcoded transformations for novel-view synthesis in view space
        #vp_rmat = torch.from_numpy(cv2.Rodrigues(np.asarray([0,np.pi/3,0]))[0]).to(Rmat.device).float()
        #vp_rmat = torch.from_numpy(cv2.Rodrigues(np.asarray([0,0.,0]))[0]).to(Rmat.device).float()
        #vp_rmat = vp_rmat[None, :, :].repeat((Rmat.shape[0], 1, 1))
        #Rmat = torch.bmm(vp_rmat, Rmat)                             # (N, 3, 3)
        #Tmat = torch.bmm(vp_rmat, Tmat[..., None])[..., 0]          # (N, 3)
        #Tmat[:, 0] -= 0.1
        #Tmat[:, 2] += 0.1

        #hardcoded transformations for novel-view synthesis in world space
        #Rmat_trans = torch.permute(Rmat, (0, 2, 1))
        #Tmat_trans = -torch.bmm(Rmat_trans, Tmat[..., None])
        #np.pi/12
        #vp_rmat = torch.from_numpy(cv2.Rodrigues(np.asarray([0,np.pi/18,0]))[0]).to(Rmat.device).float()
        #vp_rmat = vp_rmat[None, :, :].repeat((Rmat.shape[0], 1, 1))
        #Rmat_trans = torch.bmm(vp_rmat, Rmat_trans)                            # (N, 3, 3)
        #Tmat_trans = torch.bmm(vp_rmat, Tmat_trans)                            # (N, 3, 1)
        #Rmat = torch.permute(Rmat_trans, (0, 2, 1))
        #Tmat = -torch.bmm(Rmat, Tmat_trans)[..., 0]
        
        # hardcoded transformation for camera-view reconstruction (since we used a rendering size of 360, the cam_scale = 360 / 960)
        #Kinv = K2inv(0.375 * rtk[:,3])

        # code for original prepare_ray_cams
        Kmat = K2mat(rtk[:,3,:])

        Kaug = K2inv(kaug) # p = Kaug Kmat P
        Kinv = Kmatinv(Kaug.matmul(Kmat))

        return Rmat, Tmat, Kinv

    #############################################################
    ################ modified by Chonghyuk Song #################
    #def sample_pxs(self, bs, nsample, Rmat, Tmat, Kinv,
    #               dataid, frameid, frameid_sub, embedid, lineid,errid,
    #               imgs, masks, vis2d, flow, occ, dp_feats):
    def sample_pxs_objs(self, bs, nsample, Rmat_objs, Tmat_objs, Kinv_objs):
    #############################################################
    #############################################################
        """
        make sure self. is not modified
        xys:    bs, nsample, 2
        rand_inds: bs, nsample
        """
        opts = self.opts_list[-1]           # any object agnostic settings (e.g. training settings) will be taken henceforth from the background opts (index = -1)
        lineid = self.objs[-1].lineid

        # sample 1x points, sample 4x points for further selection
        '''
        nsample_a = 4*nsample
        rand_inds, xys = sample_xy(self.img_size, bs, nsample+nsample_a, self.device,
                               return_all= not(self.training), lineid=lineid)
        '''
        bs_shared = copy.copy(bs)
        nsample_shared = copy.copy(nsample)
        nsample_a_shared = 4*nsample_shared
        rand_inds_shared, xys_shared = sample_xy(self.img_size, bs_shared, nsample+nsample_a_shared, self.device,
                               return_all= not(self.training), lineid=lineid)

        '''
        # TODO: find a way to deal with the opts.use_unc flag (we currently use_unc for fg but not bkgd)
        if self.training and opts.use_unc and \
                self.progress >= (opts.warmup_steps):
            is_active=True
            nsample_s = int(opts.nactive * nsample)  # active 
            nsample   = int(nsample*(1-opts.nactive)) # uniform
        else:
            is_active=False
        '''
        # let's turn on active sampling if we find at least one object that has a 'nerf_unc'
        # then we'll use the first 'nerf_unc' we encounter to compute unc_pred, which will be used to prioritize active samples
        # assume that when we use opts.use_unc in multi-object setting, we set use_unc for all opts in self.opts_list
        #if self.training and opts.use_unc and \    

        # we will now set use_unc flag on an obj-by-obj basis, as opposed to setting it unilaterally for all objects (e.g. we’ll disable use_unc for the bkgd, and focus on finetuning the foreground)
        if self.training and np.any(np.array([opts.use_unc for opts in self.opts_list])) and \
                self.progress >= (opts.warmup_steps):           
            is_active=True
            nsample_s_shared = int(opts.nactive * nsample_shared)  # active 
            nsample_shared   = int(nsample_shared*(1-opts.nactive)) # uniform
        else:
            is_active=False

        #############################################################
        ################ modified by Chonghyuk Song #################
        # single-fg-obj case
        """
        for obj_index, obj in enumerate(self.objs):
            if 'nerf_unc' in obj.nerf_models.keys():
                nerf_unc_shared = obj.nerf_unc
                obj_nerf_unc = obj
                obj_index_nerf_unc = obj_index
                break
        """

        # multi-fg_obj case
        nerf_unc_objs = {}
        objs_nerf_unc = {}

        for obj_index, obj in enumerate(self.objs):
            # we will now set use_unc flag on an obj-by-obj basis, as opposed to setting it unilaterally for all objects (e.g. we’ll disable use_unc for the bkgd, and focus on finetuning the foreground)
            if self.opts_list[obj_index].use_unc:
                nerf_unc_objs[obj_index] = obj.nerf_unc
                objs_nerf_unc[obj_index] = obj

        num_nerf_uncs = len(nerf_unc_objs)
        #############################################################
        #############################################################

        if self.training:
            #rand_inds_a, xys_a = rand_inds[:,-nsample_a:].clone(), xys[:,-nsample_a:].clone()
            #rand_inds, xys     = rand_inds[:,:nsample].clone(), xys[:,:nsample].clone()
            rand_inds_a_shared, xys_a_shared = rand_inds_shared[:,-nsample_a_shared:].clone(), xys_shared[:,-nsample_a_shared:].clone()
            rand_inds_shared, xys_shared     = rand_inds_shared[:,:nsample_shared].clone(), xys_shared[:,:nsample_shared].clone()

        #############################################################
        ################ modified by Chonghyuk Song #################
        # single-fg-obj case
        """
        # running uncertainty estimation for one fg object
        if is_active:
            frameid_sub_in = obj_nerf_unc.frameid_sub.clone()
            dataid_in = obj_nerf_unc.dataid.clone()
            Kinv_in = Kinv_objs[obj_index_nerf_unc].clone()
            with torch.no_grad():
                # run uncertainty estimation
                #ts = frameid_sub_in.to(self.device) / self.max_ts * 2 -1
                ts = frameid_sub_in.to(self.device) / obj_nerf_unc.max_ts * 2 -1
                ts = ts[:,None,None].repeat(1,nsample_a_shared,1)
                dataid_in = dataid_in.long().to(self.device)
                #vid_code = self.vid_code(dataid_in)[:,None].repeat(1,nsample_a,1)
                vid_code = obj_nerf_unc.vid_code(dataid_in)[:,None].repeat(1,nsample_a_shared,1)

                # convert to normalized coords
                xysn = torch.cat([xys_a_shared, torch.ones_like(xys_a_shared[...,:1])],2)
                xysn = xysn.matmul(Kinv_in.permute(0,2,1))[...,:2]

                xyt = torch.cat([xysn, ts],-1)
                #xyt_embedded = self.embedding_xyz(xyt)
                xyt_embedded = obj_nerf_unc.embedding_xyz(xyt)
                xyt_code = torch.cat([xyt_embedded, vid_code],-1)
                #unc_pred = self.nerf_unc(xyt_code)[...,0]
                unc_pred_shared = nerf_unc_shared(xyt_code)[...,0]
        """

        # multi-fg_obj case
        if is_active:
            unc_pred_objs = {}

            for obj_index_nerf_unc, obj_nerf_unc in objs_nerf_unc.items():
                nerf_unc_obj = nerf_unc_objs[obj_index_nerf_unc]

                frameid_sub_in = obj_nerf_unc.frameid_sub.clone()
                dataid_in = obj_nerf_unc.dataid.clone()
                Kinv_in = Kinv_objs[obj_index_nerf_unc].clone()
                with torch.no_grad():
                    # run uncertainty estimation
                    #ts = frameid_sub_in.to(self.device) / self.max_ts * 2 -1
                    ts = frameid_sub_in.to(self.device) / obj_nerf_unc.max_ts * 2 -1
                    ts = ts[:,None,None].repeat(1,nsample_a_shared,1)
                    dataid_in = dataid_in.long().to(self.device)
                    #vid_code = self.vid_code(dataid_in)[:,None].repeat(1,nsample_a,1)
                    vid_code = obj_nerf_unc.vid_code(dataid_in)[:,None].repeat(1,nsample_a_shared,1)

                    # convert to normalized coords
                    xysn = torch.cat([xys_a_shared, torch.ones_like(xys_a_shared[...,:1])],2)
                    xysn = xysn.matmul(Kinv_in.permute(0,2,1))[...,:2]

                    xyt = torch.cat([xysn, ts],-1)
                    #xyt_embedded = self.embedding_xyz(xyt)
                    xyt_embedded = obj_nerf_unc.embedding_xyz(xyt)
                    xyt_code = torch.cat([xyt_embedded, vid_code],-1)
                    #unc_pred = self.nerf_unc(xyt_code)[...,0]
                    unc_pred_obj = nerf_unc_obj(xyt_code)[...,0]
                    unc_pred_objs[obj_index_nerf_unc] = unc_pred_obj
        #############################################################
        #############################################################

        rays_objs = []

        # compute rays for each object in their respective canonical space
        for obj_index, obj in enumerate(self.objs):
            
            xys = xys_shared
            rand_inds = rand_inds_shared
            nsample = nsample_shared
            nsample_a = nsample_a_shared
            bs = bs_shared

            if self.training:
                xys_a = xys_a_shared
                rand_inds_a = rand_inds_a_shared                    # indices of extra sampled pixels used to compute the unc_pred, then out of the (nsample_shared_a) sampled pixels, we take the top (nsample_s) pixels that have the highest uncertainty prediction

            if is_active:
                nsample_s = nsample_s_shared / num_nerf_uncs        # = int(opts.nactive * nsample_shared) / num_nerf_uncs, which represents the number of active samples per object on which we're performing the active samples

            dataid = obj.dataid
            frameid = obj.frameid
            frameid_sub = obj.frameid_sub
            embedid = obj.embedid
            lineid = obj.lineid
            errid = obj.errid

            imgs = obj.imgs
            masks = obj.masks
            deps = obj.deps
            confs = obj.confs
            vis2d = obj.vis2d
            flow = obj.flow
            occ = obj.occ
            dp_feats = obj.dp_feats
            Rmat = Rmat_objs[obj_index]
            Tmat = Tmat_objs[obj_index]
            Kinv = Kinv_objs[obj_index]

            Kinv_in=Kinv.clone()
            dataid_in=dataid.clone()
            frameid_sub_in = frameid_sub.clone()

            if self.training:

                if opts.lineload:
                    # expand frameid, Rmat,Tmat, Kinv
                    frameid_a=        frameid[:,None].repeat(1,nsample_a)
                    frameid_sub_a=frameid_sub[:,None].repeat(1,nsample_a)
                    dataid_a=          dataid[:,None].repeat(1,nsample_a)
                    errid_a=            errid[:,None].repeat(1,nsample_a)
                    Rmat_a = Rmat[:,None].repeat(1,nsample_a,1,1)
                    Tmat_a = Tmat[:,None].repeat(1,nsample_a,1)
                    Kinv_a = Kinv[:,None].repeat(1,nsample_a,1,1)
                    # expand         
                    frameid =         frameid[:,None].repeat(1,nsample)
                    frameid_sub = frameid_sub[:,None].repeat(1,nsample)
                    dataid =           dataid[:,None].repeat(1,nsample)
                    errid =             errid[:,None].repeat(1,nsample)
                    Rmat = Rmat[:,None].repeat(1,nsample,1,1)
                    Tmat = Tmat[:,None].repeat(1,nsample,1)
                    Kinv = Kinv[:,None].repeat(1,nsample,1,1)

                    batch_map   = torch.Tensor(range(bs)).to(self.device)[:,None].long()
                    batch_map_a = batch_map.repeat(1,nsample_a)
                    batch_map   = batch_map.repeat(1,nsample)

            # importance sampling
            if is_active:
                '''
                with torch.no_grad():
                    # run uncertainty estimation
                    #ts = frameid_sub_in.to(self.device) / self.max_ts * 2 -1
                    ts = frameid_sub_in.to(self.device) / obj.max_ts * 2 -1
                    ts = ts[:,None,None].repeat(1,nsample_a,1)
                    dataid_in = dataid_in.long().to(self.device)
                    #vid_code = self.vid_code(dataid_in)[:,None].repeat(1,nsample_a,1)
                    vid_code = obj.vid_code(dataid_in)[:,None].repeat(1,nsample_a,1)

                    # convert to normalized coords
                    xysn = torch.cat([xys_a, torch.ones_like(xys_a[...,:1])],2)
                    xysn = xysn.matmul(Kinv_in.permute(0,2,1))[...,:2]

                    xyt = torch.cat([xysn, ts],-1)
                    #xyt_embedded = self.embedding_xyz(xyt)
                    xyt_embedded = obj.embedding_xyz(xyt)
                    xyt_code = torch.cat([xyt_embedded, vid_code],-1)
                    #unc_pred = self.nerf_unc(xyt_code)[...,0]
                    unc_pred = obj.nerf_unc(xyt_code)[...,0]
                '''

                # preprocess to format 2,bs,w
                if opts.lineload:
                    xys =     xys.view(2,-1,2)
                    xys_a = xys_a.view(2,-1,2)                      # shape = (2, 6144, 2)
                    rand_inds =     rand_inds.view(2,-1)
                    rand_inds_a = rand_inds_a.view(2,-1)
                    frameid   =   frameid.view(2,-1)
                    frameid_a = frameid_a.view(2,-1)
                    frameid_sub   =   frameid_sub.view(2,-1)
                    frameid_sub_a = frameid_sub_a.view(2,-1)
                    dataid   =   dataid.view(2,-1)
                    dataid_a = dataid_a.view(2,-1)
                    errid   =     errid.view(2,-1)
                    errid_a   = errid_a.view(2,-1)
                    batch_map   =   batch_map.view(2,-1)
                    batch_map_a = batch_map_a.view(2,-1)
                    Rmat   =   Rmat.view(2,-1,3,3)
                    Rmat_a = Rmat_a.view(2,-1,3,3)
                    Tmat   =   Tmat.view(2,-1,3)
                    Tmat_a = Tmat_a.view(2,-1,3)
                    Kinv   =   Kinv.view(2,-1,3,3)
                    Kinv_a = Kinv_a.view(2,-1,3,3)

                    nsample_s = nsample_s * bs//2
                    bs=2

                    # merge top nsamples
                    # single-fg-obj case
                    """
                    unc_pred = unc_pred_shared.view(2,-1)
                    topk_samp = unc_pred.topk(nsample_s,dim=-1)[1] # bs,nsamp
                    # use the first imgs (in a pair) sampled index
                    xys_a =       torch.stack(          [xys_a[i][topk_samp[0]] for i in range(bs)],0)
                    rand_inds_a = torch.stack(    [rand_inds_a[i][topk_samp[0]] for i in range(bs)],0)
                    frameid_a =       torch.stack(  [frameid_a[i][topk_samp[0]] for i in range(bs)],0)
                    frameid_sub_a=torch.stack(  [frameid_sub_a[i][topk_samp[0]] for i in range(bs)],0)
                    dataid_a =         torch.stack(  [dataid_a[i][topk_samp[0]] for i in range(bs)],0)
                    errid_a =           torch.stack(  [errid_a[i][topk_samp[0]] for i in range(bs)],0)
                    batch_map_a =   torch.stack(  [batch_map_a[i][topk_samp[0]] for i in range(bs)],0)
                    Rmat_a =             torch.stack(  [Rmat_a[i][topk_samp[0]] for i in range(bs)],0)
                    Tmat_a =             torch.stack(  [Tmat_a[i][topk_samp[0]] for i in range(bs)],0)
                    Kinv_a =             torch.stack(  [Kinv_a[i][topk_samp[0]] for i in range(bs)],0)
                    """

                    # multi-fg-obj case
                    topk_samp_objs = {}
                    for obj_index_nerf_unc, unc_pred_obj in unc_pred_objs.items():
                        unc_pred_obj = unc_pred_obj.view(2,-1)                      # (bs=2, ...)
                        
                        # merge top nsamples
                        topk_samp_obj = unc_pred_obj.topk(int(nsample_s), dim=-1)[1]     # (bs=2,nsamp)      (topk outputs (topk values, topk indices) tuple, so topk[1] refers to indices)
                        topk_samp_objs[obj_index_nerf_unc] = topk_samp_obj
                    
                    # topk_samp.shape = (2, 768)
                    # topk_samp[0].shape = (768,)
                    # (BEFORE STACKING) 
                    # xys_a.shape =             (bs = 2, 6144, 2)
                    # rand_inds_a.shape =       (2, 6144) 
                    # frameid_a.shape =         (2, 6144) 
                    # frameid_sub_a.shape =     (2, 6144)
                    # dataid_a.shape =          (2, 6144)
                    # errid_a.shape =           (2, 6144)
                    # batch_map_a.shape =       (2, 6144)
                    # Rmat_a.shape =            (2, 6144, 3, 3)
                    # Tmat_a.shape =            (2, 6144, 3)
                    # Kinv_a.shape =            (2, 6144, 3, 3)

                    # (AFTER STACKING) 
                    # xys_a.shape =             (bs = 2, 768, 2)
                    # rand_inds_a.shape =       (2, 768) 
                    # frameid_a.shape =         (2, 768) 
                    # frameid_sub_a.shape =     (2, 768)
                    # dataid_a.shape =          (2, 768)
                    # errid_a.shape =           (2, 768)
                    # batch_map_a.shape =       (2, 768)
                    # Rmat_a.shape =            (2, 768, 3, 3)
                    # Tmat_a.shape =            (2, 768, 3)
                    # Kinv_a.shape =            (2, 768, 3, 3)
                    # xys.shape =               (2, 768 * 2 = 1536, 2)

                    xys_a =             torch.cat([torch.stack([xys_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)              # torch.cat(list of arrays of shape (2, 1536 / num_nerf_unc, 2))
                    rand_inds_a =       torch.cat([torch.stack([rand_inds_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)        # torch.cat(list of arrays of shape (2, 1536 / num_nerf_unc))
                    frameid_a =         torch.cat([torch.stack([frameid_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)          # '''
                    frameid_sub_a =     torch.cat([torch.stack([frameid_sub_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)      # '''                    
                    dataid_a =          torch.cat([torch.stack(  [dataid_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)         # torch.cat(list of arrays of shape (2, 1536 / num_nerf_unc))
                    errid_a =           torch.cat([torch.stack(  [errid_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)          # torch.cat(list of arrays of shape (2, 1536 / num_nerf_unc))
                    batch_map_a =       torch.cat([torch.stack(  [batch_map_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)      # torch.cat(list of arrays of shape (2, 1536 / num_nerf_unc))
                    Rmat_a =            torch.cat([torch.stack(  [Rmat_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)           # torch.cat(list of arrays of shape (2, 1536 / num_nerf_unc, 3, 3))
                    Tmat_a =            torch.cat([torch.stack(  [Tmat_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)           # torch.cat(list of arrays of shape (2, 1536 / num_nerf_unc, 3))
                    Kinv_a =            torch.cat([torch.stack(  [Kinv_a[i][topk_samp[0]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)           # torch.cat(list of arrays of shape (2, 1536 / num_nerf_unc, 3, 3))

                    # combining regular samples with active samples
                    xys =       torch.cat([xys,xys_a],1)                    
                    rand_inds = torch.cat([rand_inds,rand_inds_a],1)
                    frameid =   torch.cat([frameid,frameid_a],1)
                    frameid_sub=torch.cat([frameid_sub,frameid_sub_a],1)
                    dataid =    torch.cat([dataid,dataid_a],1)
                    errid =     torch.cat([errid,errid_a],1)
                    batch_map = torch.cat([batch_map,batch_map_a],1)
                    Rmat =      torch.cat([Rmat,Rmat_a],1)
                    Tmat =      torch.cat([Tmat,Tmat_a],1)
                    Kinv =      torch.cat([Kinv,Kinv_a],1)
                else:
                    #topk_samp = unc_pred.topk(nsample_s,dim=-1)[1] # bs,nsamp
                    # single-fg-obj case
                    """
                    topk_samp = unc_pred_shared.topk(nsample_s,dim=-1)[1] # bs,nsamp
                    
                    xys_a =       torch.stack(      [xys_a[i][topk_samp[i]] for i in range(bs)],0)
                    rand_inds_a = torch.stack([rand_inds_a[i][topk_samp[i]] for i in range(bs)],0)
                    """

                    # multi-fg-obj case
                    topk_samp_objs = {}
                    for obj_index_nerf_unc, unc_pred_obj in unc_pred_objs.items():
                        # merge top nsamples
                        topk_samp_obj = unc_pred_obj.topk(nsample_s, dim=-1)[1]     # (bs=2,nsamp)      (topk outputs (topk values, topk indices) tuple, so topk[1] refers to indices)
                        topk_samp_objs[obj_index_nerf_unc] = topk_samp_obj
                    xys_a =         torch.cat([torch.stack(      [xys_a[i][topk_samp[i]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs.items()], 1)
                    rand_inds_a =   torch.cat([torch.stack(      [rand_inds_a[i][topk_samp[i]] for i in range(bs)],0) for _, topk_samp in topk_samp_objs].items(), 1)
                    
                    xys =       torch.cat([xys,xys_a],1)
                    rand_inds = torch.cat([rand_inds,rand_inds_a],1)

            # for line: reshape to 2*bs, 1,...
            if self.training and opts.lineload:
                frameid =         frameid.view(-1)
                frameid_sub = frameid_sub.view(-1)
                dataid =           dataid.view(-1)
                errid =             errid.view(-1)
                batch_map =     batch_map.view(-1)
                xys =           xys.view(-1,1,2)
                rand_inds = rand_inds.view(-1,1)
                Rmat = Rmat.view(-1,3,3)
                Tmat = Tmat.view(-1,3)
                Kinv = Kinv.view(-1,3,3)

            #near_far = self.near_far[frameid.long()]
            near_far = obj.near_far[frameid.long()]
            rays = raycast(xys, Rmat, Tmat, Kinv, near_far)
        
            # need to reshape dataid, frameid_sub, embedid #TODO embedid equiv to frameid
            #self.update_rays(rays, bs>1, dataid, frameid_sub, frameid, xys, Kinv)
            self.update_rays_obj(obj_index, rays, bs>1, dataid, frameid_sub, frameid, xys, Kinv)
            
            #if 'bones' in self.nerf_models.keys():
            if 'bones' in obj.nerf_models.keys():
                # update delta rts fw
                #self.update_delta_rts(rays)
                self.update_delta_rts_obj(obj_index, rays)

            #############################################################
            ################ modified by Chonghyuk Song #################
            # for line: 2bs*nsamp,1
            # for batch:2bs,nsamp
            #TODO reshape imgs, masks, etc.
            if self.training and opts.lineload:
                #self.obs_to_rays_line(rays, rand_inds, imgs, masks, vis2d, flow, occ, 
                #        dp_feats, batch_map)
                self.obs_to_rays_line(obj_index, rays, rand_inds, imgs, masks, deps, confs, vis2d, flow, occ, 
                        dp_feats, batch_map)
            else:
                #self.obs_to_rays(rays, rand_inds, imgs, masks, vis2d, flow, occ, dp_feats)
                self.obs_to_rays(obj_index, rays, rand_inds, imgs, masks, deps, confs, vis2d, flow, occ, dp_feats)
            #############################################################
            #############################################################

            # TODO visualize samples
            #pdb.set_trace()
            #self.imgs_samp = []
            #for i in range(bs):
            #    self.imgs_samp.append(draw_pts(self.imgs[i], xys_a[i]))
            #self.imgs_samp = torch.stack(self.imgs_samp,0)

            rays_objs.append(rays)

        #return rand_inds, rays, frameid, errid
        return rand_inds, rays_objs, frameid, errid
    
    #############################################################
    ################ modified by Chonghyuk Song #################
    #def obs_to_rays_line(self, rays, rand_inds, imgs, masks, vis2d,
    #        flow, occ, dp_feats,batch_map):
    def obs_to_rays_line(self, obj_index, rays, rand_inds, imgs, masks, deps, confs, vis2d,
            flow, occ, dp_feats,batch_map):
    #############################################################
    #############################################################
        """
        convert imgs, masks, flow, occ, dp_feats (and depths and their corresponding confidence maps) to rays
        rand_map: map pixel index to original batch index
        rand_inds: bs, 
        """
        opts = self.opts_list[obj_index]
        rays['img_at_samp']=torch.gather(imgs[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,3,1))[:,None][...,0]
        rays['sil_at_samp']=torch.gather(masks[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,1,1))[:,None][...,0]
        #############################################################
        ################ modified by Chonghyuk Song #################
        rays['dep_at_samp']=torch.gather(deps[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,1,1))[:,None][...,0]
        rays['conf_at_samp']=torch.gather(confs[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,1,1))[:,None][...,0]
        #############################################################
        #############################################################
        rays['vis_at_samp']=torch.gather(vis2d[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,1,1))[:,None][...,0]
        rays['flo_at_samp']=torch.gather(flow[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,2,1))[:,None][...,0]
        rays['cfd_at_samp']=torch.gather(occ[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,1,1))[:,None][...,0]
        if opts.use_embed:
            rays['feats_at_samp']=torch.gather(dp_feats[batch_map][...,0], 2, 
                rand_inds[:,None].repeat(1,16,1))[:,None][...,0]

    #############################################################
    ################ modified by Chonghyuk Song ################# 
    #def obs_to_rays(self, rays, rand_inds, imgs, masks, vis2d,
    #        flow, occ, dp_feats):
    def obs_to_rays(self, obj_index, rays, rand_inds, imgs, masks, deps, confs, vis2d,
            flow, occ, dp_feats):
    #############################################################
    #############################################################
        """
        convert imgs, masks, flow, occ, dp_feats (and depths and their corresponding confidence maps) to rays
        """
        #TODO: select the correct opts from self.opts_list (need to feed in obj_index into this function)
        opts = self.opts_list[obj_index]
        bs = imgs.shape[0]
        rays['img_at_samp'] = torch.stack([imgs[i].view(3,-1).T[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,3
        rays['sil_at_samp'] = torch.stack([masks[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        #############################################################
        ################ modified by Chonghyuk Song #################
        rays['dep_at_samp'] = torch.stack([deps[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        rays['conf_at_samp'] = torch.stack([confs[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        #############################################################
        #############################################################
        rays['vis_at_samp'] = torch.stack([vis2d[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        rays['flo_at_samp'] = torch.stack([flow[i].view(2,-1).T[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,2
        rays['cfd_at_samp'] = torch.stack([occ[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        if opts.use_embed:
            feats_at_samp = [dp_feats[i].view(16,-1).T\
                             [rand_inds[i].long()] for i in range(bs)]
            feats_at_samp = torch.stack(feats_at_samp,0) # bs,ns,num_feat
            rays['feats_at_samp'] = feats_at_samp
        
    def update_delta_rts_obj(self, obj_index, rays):
        """
        change bone_rts_fw to delta fw
        """
        opts = self.opts_list[obj_index]
        obj = self.objs[obj_index]

        #bones_rst, bone_rts_rst = correct_bones(self, self.nerf_models['bones'])
        #self.nerf_models['bones_rst']=bones_rst
        bones_rst, bone_rts_rst = correct_bones(obj, obj.nerf_models['bones'])
        obj.nerf_models['bones_rst']=bones_rst

        # delta rts
        rays['bone_rts'] = correct_rest_pose(opts, rays['bone_rts'], bone_rts_rst)

        if 'bone_rts_target' in rays.keys():       
            rays['bone_rts_target'] = correct_rest_pose(opts, 
                                            rays['bone_rts_target'], bone_rts_rst)

        if 'bone_rts_dentrg' in rays.keys():
            rays['bone_rts_dentrg'] = correct_rest_pose(opts, 
                                            rays['bone_rts_dentrg'], bone_rts_rst)

    def update_rays_obj(self, obj_index, rays, is_pair, dataid, frameid_sub, embedid, xys, Kinv):
        """
        """
        opts = self.opts_list[obj_index]
        obj = self.objs[obj_index]

        # append target frame rtk
        embedid = embedid.long().to(self.device)
        if is_pair:
            rtk_vec = rays['rtk_vec'] # bs, N, 21
            rtk_vec_target = rtk_vec.view(2,-1).flip(0)
            rays['rtk_vec_target'] = rtk_vec_target.reshape(rays['rtk_vec'].shape)
            
            embedid_target = embedid.view(2,-1).flip(0).reshape(-1)
            if opts.flowbw:
                #time_embedded_target = self.pose_code(embedid_target)[:,None]
                time_embedded_target = obj.pose_code(embedid_target)[:,None]
                rays['time_embedded_target'] = time_embedded_target.repeat(1,
                                                            rays['nsample'],1)
            #elif opts.lbs and self.num_bone_used>0:
            elif opts.lbs and obj.num_bone_used>0:
                #bone_rts_target = self.nerf_body_rts(embedid_target)
                bone_rts_target = obj.nerf_body_rts(embedid_target)
                rays['bone_rts_target'] = bone_rts_target.repeat(1,rays['nsample'],1)

        # pass time-dependent inputs
        #time_embedded = self.pose_code(embedid)[:,None]
        time_embedded = obj.pose_code(embedid)[:,None]
        rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
        #if opts.lbs and self.num_bone_used>0:
        if opts.lbs and obj.num_bone_used>0:
            #bone_rts = self.nerf_body_rts(embedid)
            bone_rts = obj.nerf_body_rts(embedid)
            rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)

        if opts.env_code:
            #rays['env_code'] = self.env_code(embedid)[:,None]
            rays['env_code'] = obj.env_code(embedid)[:,None]
            rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)
            #rays['env_code'] = self.env_code(dataid.long().to(self.device))
            #rays['env_code'] = rays['env_code'][:,None].repeat(1,rays['nsample'],1)

        if opts.use_unc:
            #ts = frameid_sub.to(self.device) / self.max_ts * 2 -1
            ts = frameid_sub.to(self.device) / obj.max_ts * 2 -1
            ts = ts[:,None,None].repeat(1,rays['nsample'],1)
            rays['ts'] = ts
        
            dataid = dataid.long().to(self.device)
            #vid_code = self.vid_code(dataid)[:,None].repeat(1,rays['nsample'],1)
            vid_code = obj.vid_code(dataid)[:,None].repeat(1,rays['nsample'],1)
            rays['vid_code'] = vid_code
            
            xysn = torch.cat([xys, torch.ones_like(xys[...,:1])],2)
            xysn = xysn.matmul(Kinv.permute(0,2,1))[...,:2]
            rays['xysn'] = xysn