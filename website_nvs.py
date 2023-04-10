import os, sys, pdb, pickle, json
from glob import glob
import numpy as np
from tqdm import tqdm
import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br, video
import shutil
import imageio
import argparse
import configparser
import datetime

#WIDTH = 600
WIDTH = 900
#nvs_types = ["3d filter", "input-view recon.", "egocentric-view", "third-person-view", "bird's eye view", "stereo-view", "fixed-view", "obj. and cam. poses"]
#nvs_types = ["3d filter", "input-view recon.", "egocentric-view", "third-person-view", "fixed-view", "bird's eye view", "turntable (frozen time)", "obj. and cam. poses"]
nvs_types = ["input-view recon.", "input-view-rgberror", "input-view-dpherror", "stereo-view", "stereo-view-rgberror", "stereo-view-dpherror"]
#nvs_types = ["input-view-rgberror", "input-view-dpherror",  "fixed-view"]

video_types = ["GT RGB", "Rendered RGB", "GT Depth", "Rendered Depth", "Rendered Mask", "Rendered Mask"]
video_types_stereo = ["GT RGB (1st cam)", "GT RGB (2nd cam)", "Rendered RGB", "GT Depth", "Rendered Depth", "Rendered Mask", "Rendered Mask"]
video_types_error = ["GT (2nd cam)", "Rendered", "Error"]

def header_row(nvs_types):
    header_row = tr()

    _td = td(style="word-wrap: break-word;", halign="center", valign="top", font=15)
    _td.add(p(""))
    header_row.add(_td)

    for nvs_type in nvs_types:
        
        if nvs_type == "3d filter" or nvs_type == "obj. and cam. poses":
            colspan = 1
        elif nvs_type == "stereo-view":
            colspan = 7
        elif nvs_type == "stereo-view-rgberror":
            colspan = 3
        elif nvs_type == "stereo-view-dpherror":
            colspan = 3
        elif nvs_type == "input-view-rgberror":
            colspan = 3
        elif nvs_type == "input-view-dpherror":
            colspan = 3
        else:
            colspan = 6

        _td = td(style="word-wrap: break-word;", halign="center", valign="top", font=15, colspan = colspan)
        _td.add(p(nvs_type))
        header_row.add(_td)
    
    return header_row

def videodescriptor_row(nvs_types, video_types):
    header_row = tr()

    _td = td(style="word-wrap: break-word;", halign="center", valign="top", font=15)
    _td.add(p(""))
    header_row.add(_td)

    for nvs_type in nvs_types:
        
        if nvs_type == "3d filter" or nvs_type == "obj. and cam. poses":
            video_names = video_types[0:1]
        elif nvs_type == "stereo-view":
            video_names = video_types_stereo
        elif nvs_type.endswith("rgberror") or nvs_type.endswith("dpherror"):
            video_names = video_types_error
        else:
            video_names = video_types

        for video_name in video_names:
            _td = td(style="width:%dpx" % WIDTH, halign="center", valign="top", font=15)
            _td.add(p(video_name))
            header_row.add(_td)
    
    return header_row

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="render website")
    parser.add_argument('--seqnames', default='', help='a string of seqnames separated by spaces that specifies the experiment name')
    parser.add_argument('--rowheadings', default='', help='experiment descriptions (row headings')
    parser.add_argument('--pagename', default='', help='used to name the html file')
    args = parser.parse_args()
    basedir = os.getcwd()

    # 2. defining the experiment names (the titles for each row)
    exp_names = args.seqnames.split(" ")
    rowheadings = args.rowheadings.split("| ")

    # 3. determine the number of total videos for specified category
    num_videos = len(nvs_types)

    # 4. make document
    doc = dominate.document(title="")
    with doc: h3("")

    # 5. make table
    t_main = table(border=1, style="table-layout: fixed;")

    # 6. adding a title row to the table
    t_main.add(header_row(nvs_types))

    # 6a. add row describing each type of video

    # 7. write every row
    # exp_name is the actual name of the log file that contains the nvs resuls
    # rowheadings is the text displayed on the webpage out of consideration for viewers
    for i, (exp_name, rowheading) in enumerate(zip(exp_names, rowheadings)):
        t_main.add(videodescriptor_row(nvs_types, video_types))
        src_dir = os.path.join("../../", exp_name)

        # initialize row (td means every element in row)
        #_tr = tr(td(p(exp_name)))
        _tr = tr(td(p(rowheading)))

        for nvs_type in nvs_types:
            #nvs_types = ["obj. and cam. poses" "input-view recon.", "bird's eye view", "turntable (frozen time)", "fixed-view"]

            if nvs_type == "obj. and cam. poses":
                video_name = "objpose.mp4"
                video_width = WIDTH / 6
                colspan = 1
            if nvs_type == "3d filter":
                video_name = "nvs-inputview-rgb_with_asset.mp4"
                video_width = WIDTH / 6
                colspan = 1
            elif nvs_type == "input-view recon.":
                video_name = "nvs-inputview-all.mp4"
                video_width = WIDTH
                colspan = 6
            elif nvs_type == "stereo-view":
                video_name = "nvs-stereoview-all.mp4"
                video_width = WIDTH
                colspan = 7
            elif nvs_type == "stereo-view-rgberror":
                video_name = "nvs-stereoview-comparergb.mp4"
                video_width = WIDTH
                colspan = 3
            elif nvs_type == "input-view-dpherror":
                video_name = "nvs-inputview-comparedph.mp4"
                video_width = WIDTH
                colspan = 3
            elif nvs_type == "input-view-rgberror":
                video_name = "nvs-inputview-comparergb.mp4"
                video_width = WIDTH
                colspan = 3
            elif nvs_type == "stereo-view-dpherror":
                video_name = "nvs-stereoview-comparedph.mp4"
                video_width = WIDTH
                colspan = 3
            elif nvs_type == "bird's eye view":
                video_name = "nvs-bev-all.mp4"
                video_width = WIDTH
                colspan = 6
            elif nvs_type == "turntable (frozen time)":
                video_name = "nvs-frozentime-all.mp4"
                video_width = WIDTH
                colspan = 6
            elif nvs_type == "fixed-view":
                video_name = "nvs-fixedview-all.mp4"
                video_width = WIDTH
                colspan = 6
            elif nvs_type == "stereo-view":
                video_name = "nvs-stereoview-all.mp4"
                video_width = WIDTH    
                colspan = 6
            elif nvs_type == "egocentric-view":
                video_name = "nvs-fpsview-all.mp4"
                video_width = WIDTH    
                colspan = 6
            elif nvs_type == "third-person-view":
                video_name = "nvs-tpsview-all.mp4"
                video_width = WIDTH
                colspan = 6
            src_file = os.path.join(src_dir, video_name)
            
            _td = td(style="word-wrap: break-word;", halign="center", valign="top", colspan=colspan)
            _td.add(video(style="width:%dpx" % video_width, src=src_file, autoplay=True, controls=True, loop = True))
            _tr.add(_td)

        # add the row to the table
        t_main.add(_tr)
    
    # 8. render the table "t_main" into a folder for that week's monday meeting
    # make a folder for that week's meeting
    today = datetime.date.today()
    next_monday = str(today + datetime.timedelta(days=-today.weekday(), weeks=0))
    
    webpages_dir = "{}/logdir/webpages/{}".format(basedir, next_monday)
    if not os.path.exists(webpages_dir):
        os.makedirs(webpages_dir)
    # copy the 'make_webpages.sh' script to the folder
    shutil.copyfile("{}/make_webpages.sh".format(basedir), "{}/make_webpages.sh".format(webpages_dir)) 

    # render the webpage
    with open(os.path.join(webpages_dir, "{}.html".format(args.pagename)), "w") as f: f.write(t_main.render())
