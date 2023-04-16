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

    #print("[ENTIRE] rms rgb error: {}".format(rms_rgb_error_entire))
    print("[ENTIRE] avg lpips: {}".format(np.round(avg_lpips_entire, 3)))
    print("[ENTIRE] avg psnr: {}".format(np.round(avg_psnr_entire, 2)))
    print("[ENTIRE] avg ssim: {}".format(np.round(avg_ssim_entire, 3)))
    print("[ENTIRE] rms depth error: {}".format(np.round(rms_depth_error_entire, 4)))
    print("[ENTIRE] avg depth acc at 10cm: {}".format(np.round(rms_depth_acc_at_10cm_entire, 4)))
    print("\n")