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
nvs_types = ["obj. and cam. poses"]

def header_row(nvs_types):
    header_row = tr()

    _td = td(style="word-wrap: break-word;", halign="center", valign="top", font=15)
    _td.add(p(""))
    header_row.add(_td)

    for nvs_type in nvs_types:
        _td = td(style="word-wrap: break-word;", halign="center", valign="top", font=15)
        _td.add(p(nvs_type))
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

    # 7. write every row
    # exp_name is the actual name of the log file that contains the nvs resuls
    # rowheadings is the text displayed on the webpage out of consideration for viewers
    for i, (exp_name, rowheadings) in enumerate(zip(exp_names, rowheadings)):
        src_dir = os.path.join("../../", exp_name)

        # initialize row (td means every element in row)
        #_tr = tr(td(p(exp_name)))
        _tr = tr(td(p(rowheadings)))

        for nvs_type in nvs_types:
            #nvs_types = ["obj. and cam. poses" "input-view recon.", "bird's eye view", "turntable (frozen time)", "fixed-view"]

            if nvs_type == "obj. and cam. poses":
                video_name = "objpose-all.mp4"
                video_width = WIDTH / 5 * 2
            elif nvs_type == "input-view recon.":
                video_name = "nvs-inputview-all.mp4"
                video_width = WIDTH
            elif nvs_type == "bird's eye view":
                video_name = "nvs-bev-all.mp4"
                video_width = WIDTH
            elif nvs_type == "turntable (frozen time)":
                video_name = "nvs-frozentime-all.mp4"
                video_width = WIDTH
            elif nvs_type == "fixed-view":
                video_name = "nvs-fixedview-all.mp4"
                video_width = WIDTH
            elif nvs_type == "stereo-view":
                video_name = "nvs-stereoview-all.mp4"
                video_width = WIDTH    
            elif nvs_type == "egocentric-view":
                video_name = "nvs-fpsview-all.mp4"
                video_width = WIDTH    
            elif nvs_type == "third-person-view":
                video_name = "nvs-tpsview-all.mp4"
                video_width = WIDTH
            src_file = os.path.join(src_dir, video_name)
            
            _td = td(style="word-wrap: break-word;", halign="center", valign="top")
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
