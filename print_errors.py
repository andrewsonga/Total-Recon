import numpy as np
import glob
import argparse
from nnutils.eval_utils import rms_metric_over_allframes, average_metric_over_allframes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="print metrics")
    parser.add_argument('--seqname', default='', help='dir name under which the metrics are stored')
    parser.add_argument('--view', default='stereoview', help='view from which we want to print the metrics (input-view or stereo-view)')
    args = parser.parse_args()

    checkpoint_dir = "logdir"
    seqname = args.seqname
    view = args.view
    obj_dirs = sorted(glob.glob("{}/{}/obj[0-9]".format(checkpoint_dir, seqname)))

    # rms rgb error (entire)
    rms_rgb_error_entire = np.load('logdir/{}/nvs-{}-rmsrgberror.npy'.format(seqname, view))

    # mean psnr (entire)
    avg_psnr_entire = np.load('logdir/{}/nvs-{}-psnr.npy'.format(seqname, view))

    # mean ssim (entire)
    avg_ssim_entire = np.load('logdir/{}/nvs-{}-ssim.npy'.format(seqname, view))

    # mean lpips (entire)
    avg_lpips_entire = np.load('logdir/{}/nvs-{}-lpips.npy'.format(seqname, view))

    # rms depth error (entire)
    rms_depth_error_entire = np.load('logdir/{}/nvs-{}-rmsdeptherror.npy'.format(seqname, view))

    # mean depth accuracy (entire)
    rms_depth_acc_at_10cm_entire = np.load('logdir/{}/nvs-{}-depaccat10cm.npy'.format(seqname, view))

    # mean cd (entire)
    avg_cd_entire = np.load('logdir/{}/nvs-{}-cd.npy'.format(seqname, view))

    # mean f-score @ 5cm (entire)
    avg_fat5cm_entire = np.load('logdir/{}/nvs-{}-fat5cm.npy'.format(seqname, view))

    # mean f-score @ 10cm (entire)
    avg_fat10cm_entire = np.load('logdir/{}/nvs-{}-fat10cm.npy'.format(seqname, view))

    #print("[ENTIRE] rms rgb error: {}".format(rms_rgb_error_entire))
    print("[ENTIRE] avg lpips: {}".format(np.round(avg_lpips_entire, 3)))
    print("[ENTIRE] avg psnr: {}".format(np.round(avg_psnr_entire, 2)))
    print("[ENTIRE] avg ssim: {}".format(np.round(avg_ssim_entire, 3)))
    print("[ENTIRE] rms depth error: {}".format(np.round(rms_depth_error_entire, 4)))
    print("[ENTIRE] avg depth acc at 10cm: {}".format(np.round(rms_depth_acc_at_10cm_entire, 4)))
    print("[ENTIRE] avg cd: {}".format(np.round(avg_cd_entire, 4)))
    print("[ENTIRE] avg f-score @ 5cm: {}".format(np.round(avg_fat5cm_entire, 3)))
    print("[ENTIRE] avg f-score @ 10cm : {}".format(np.round(avg_fat10cm_entire, 3)))
    print("\n")

    """
    # per-object metrics
    rms_rgb_error_objs = []
    rms_depth_error_objs = []
    for obj_index in range(len(obj_dirs)):
        # rms rgb error (obj)
        rms_rgb_error_obj = np.load('logdir/{}/nvs-{}-rmsrgberror-obj{}.npy'.format(seqname, view, obj_index))

        # mean psnr (obj)
        avg_psnr_obj = np.load('logdir/{}/nvs-{}-psnr-obj{}.npy'.format(seqname, view, obj_index))

        # mean ssim (obj)
        avg_ssim_obj = np.load('logdir/{}/nvs-{}-ssim-obj{}.npy'.format(seqname, view, obj_index))

        # mean lpips (obj)
        avg_lpips_obj = np.load('logdir/{}/nvs-{}-lpips-obj{}.npy'.format(seqname, view, obj_index))

        # rms depth error (obj)
        rms_depth_error_obj = np.load('logdir/{}/nvs-{}-rmsdeptherror-obj{}.npy'.format(seqname, view, obj_index))

        # mean cd (obj)
        avg_cd_obj = np.load('logdir/{}/nvs-{}-cd-obj{}.npy'.format(seqname, view, obj_index))

        # mean f-score @ 5cm (obj)
        avg_fat5cm_obj = np.load('logdir/{}/nvs-{}-fat5cm-obj{}.npy'.format(seqname, view, obj_index))

        # mean f-score @ 10cm (obj)
        avg_fat10cm_obj = np.load('logdir/{}/nvs-{}-fat10cm-obj{}.npy'.format(seqname, view, obj_index))

        #print("[OBJ {}] rms rgb error: {}".format(obj_index, rms_rgb_error_obj))
        print("[OBJ {}] avg lpips: {}".format(obj_index, np.round(avg_lpips_obj, 3)))
        print("[OBJ {}] avg psnr: {}".format(obj_index, np.round(avg_psnr_obj, 2)))
        print("[OBJ {}] avg ssim: {}".format(obj_index, np.round(avg_ssim_obj, 3)))
        print("[OBJ {}] rms depth error: {}".format(obj_index, np.round(rms_depth_error_obj, 4)))
        print("[OBJ {}] avg cd: {}".format(obj_index, np.round(avg_cd_obj, 4)))
        print("[OBJ {}] avg f-score @ 5cm: {}".format(obj_index, np.round(avg_fat5cm_obj, 3)))
        print("[OBJ {}] avg f-score @ 10cm : {}".format(obj_index, np.round(avg_fat10cm_obj, 3)))
        print("\n")
    """
    
    # per-object metrics
    rms_rgb_error_objs = []
    rms_depth_error_objs = []
    for obj_index in range(len(obj_dirs)):
        # rms rgb error (obj)
        # rms_rgb_error_obj = np.load('logdir/{}/nvs-{}-rmsrgberror-obj{}.npy'.format(seqname, view, obj_index))

        # mean psnr (obj)
        avg_psnr_obj = np.load('logdir/{}/nvs-{}-psnr-pixelavg-obj{}.npy'.format(seqname, view, obj_index))

        # mean ssim (obj)
        avg_ssim_obj = np.load('logdir/{}/nvs-{}-ssim-pixelavg-obj{}.npy'.format(seqname, view, obj_index))

        # mean lpips (obj)
        avg_lpips_obj = np.load('logdir/{}/nvs-{}-lpips-pixelavg-obj{}.npy'.format(seqname, view, obj_index))

        # rms depth error (obj)
        rms_depth_error_obj = np.load('logdir/{}/nvs-{}-rmsdeptherror-pixelavg-obj{}.npy'.format(seqname, view, obj_index))

        # rms depth error (obj)
        avg_depth_acc_obj = np.load('logdir/{}/nvs-{}-depaccat10cm-pixelavg-obj{}.npy'.format(seqname, view, obj_index))

        # mean cd (obj)
        #avg_cd_obj = np.load('logdir/{}/nvs-{}-cd-obj{}.npy'.format(seqname, view, obj_index))

        # mean f-score @ 5cm (obj)
        #avg_fat5cm_obj = np.load('logdir/{}/nvs-{}-fat5cm-obj{}.npy'.format(seqname, view, obj_index))

        # mean f-score @ 10cm (obj)
        #avg_fat10cm_obj = np.load('logdir/{}/nvs-{}-fat10cm-obj{}.npy'.format(seqname, view, obj_index))

        #print("[OBJ {}] rms rgb error: {}".format(obj_index, rms_rgb_error_obj))
        print("[OBJ {}] avg lpips: {}".format(obj_index, np.round(avg_lpips_obj, 3)))
        print("[OBJ {}] avg psnr: {}".format(obj_index, np.round(avg_psnr_obj, 2)))
        print("[OBJ {}] avg ssim: {}".format(obj_index, np.round(avg_ssim_obj, 3)))
        print("[OBJ {}] rms depth error: {}".format(obj_index, np.round(rms_depth_error_obj, 4)))
        print("[OBJ {}] avg depth acc @ 10cm: {}".format(obj_index, np.round(avg_depth_acc_obj, 4)))
        print("\n")

    #rgb_errors_obj0 = np.load('logdir/{}/nvs-{}-rgberrors-obj0.npy'.format(seqname, view))
    #rgb_errors_obj1 = np.load('logdir/{}/nvs-{}-rgberrors-obj1.npy'.format(seqname, view))

    #nonempty_frames_obj0 = np.logical_not(np.isnan(np.stack(rgb_errors_obj0)))            # frames with valid masks
    #nonempty_frames_obj1 = np.logical_not(np.isnan(np.stack(rgb_errors_obj1)))            # frames with valid masks

    #print("[ENTIRE] avg lpips: {}".format(average_metric_over_allframes(lpips_all[nonempty_frames_obj1])))
    #print("[ENTIRE] avg psnr: {}".format(average_metric_over_allframes(psnr_all[nonempty_frames_obj1])))
    #print("[ENTIRE] avg ssim: {}".format(average_metric_over_allframes(ssim_all[nonempty_frames_obj1])))
    #print("[ENTIRE] rms depth error: {}".format(rms_metric_over_allframes(depth_error_all[[nonempty_frames_obj1]])))