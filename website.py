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

WIDTH = 120
start_frames = [0, 440, 675, 1005]
end_frames = [439, 674, 1004, 1503]

# 1. defining the file names for each column (each type of visualization)
vis_types = ["reconstruction", "novel view synthesis", "camera poses"]

def header_row(num_videos):
    header_row = tr()

    _td = td(style="word-wrap: break-word;", halign="center", valign="top", font=15)
    _td.add(p(""))
    header_row.add(_td)

    for i in range(num_videos):
        _td = td(colspan="3", style="word-wrap: break-word;", halign="center", valign="top", font=15)
        _td.add(p("video%03d"%i))
        header_row.add(_td)
    
    return header_row

def title_row(num_videos):
    title_row = tr()

    _td = td(style="word-wrap: break-word;", halign="center", valign="top", font=15)
    _td.add(p(""))
    title_row.add(_td)
    
    for _ in range(num_videos):
        for vis_type in vis_types:
            _td = td(style="word-wrap: break-word;", halign="center", valign="top", font=15)
            _td.add(p(vis_type))
            title_row.add(_td)
    
    return title_row

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="render website")
    parser.add_argument('--seqnames', default='', help='a string of seqnames separated by spaces that specifies the experiment name')
    parser.add_argument('--rowheadings', default='', help='experiment descriptions (row headings')
    parser.add_argument('--vidids', default='', help=' a string of vidid used to specify which video to reconstruct e.g. "0 0-1-2-3 1-2 0-1-2-3-4-5-6-7"')
    parser.add_argument('--pagename', default='', help='used to name the html file')
    args = parser.parse_args()

    # 2. defining the experiment names (the titles for each row)
    exp_names = args.seqnames.split(" ")
    rowheadings = args.rowheadings.split("| ")

    # 3. defining the vidids to include inside the table
    vidids = args.vidids.split(" ")

    # 3. determine the number of total videos for specified category
    if exp_names[0].startswith("cat-pikachiu-rgbd"):
        num_videos = 4
    elif exp_names[0].startswith("andrew"):
        num_videos = 7

    # 4. make document
    doc = dominate.document(title="")
    with doc: h3("")

    # 5. make table
    t_main = table(border=1, style="table-layout: fixed;")

    # 6. adding a title row to the table
    t_main.add(header_row(num_videos))
    t_main.add(title_row(num_videos))

    # 7. write every row
    for i, (exp_name, rowheadings) in enumerate(zip(exp_names, rowheadings)):
        src_dir = os.path.join("logdir", exp_name + "-e120-b256-ft3")

        # initialize row (td means every element in row)
        #_tr = tr(td(p(exp_name)))
        _tr = tr(td(p(rowheadings)))

        # read the config file to find for which videos we want to insert visualizations
        config = configparser.RawConfigParser()
        config.read('configs/%s.config'%exp_name)
        
        data_ids = vidids[i].split("-")        # data ids separate by "-"

        vid_indices = []
        for j in data_ids:
            vid_index = int(config.get('data_%s'%j, 'datapath').split("/")[-2][-3:])
            vid_indices.append(vid_index)

        # write or skip each element in row
        index_so_far = 0
        for data_id, vid_index in enumerate(vid_indices):

            # populate elements up to vid_index with blanks
            for _ in range(index_so_far, vid_index):
                for vis_type in vis_types:
                    if vis_type == "reconstruction":
                        width = 4 * WIDTH
                    elif vis_type == "novel view synthesis":
                        width = 3 * WIDTH
                    elif vis_type == "camera poses":
                        width = 3 * WIDTH
                    _td = td(style="word-wrap: break-word;", halign="center", valign="top")
                    _td.add(img(style="width:%dpx" % width))                                    # insert blank image into td
                    _tr.add(_td)

            # fillup cells corresponding to vid_index with the visualizations
            for vis_type in vis_types:
                if vis_type == "reconstruction":
                    src_file = os.path.join(src_dir, "nvs-{}-{}-traj-all.gif".format(data_id, data_id))
                    width = 4 * WIDTH
                elif vis_type == "novel view synthesis":
                    src_file = os.path.join(src_dir, exp_name + "-{%d}-all.gif"%data_id)
                    width = 3 * WIDTH
                elif vis_type == "camera poses":
                    src_file = os.path.join(src_dir, "mesh-cam-start%04d-end%04d.gif"%(start_frames[data_id], start_frames[data_id] + end_frames[vid_index] - start_frames[vid_index]))
                    width = 3 * WIDTH
                
                _td = td(style="word-wrap: break-word;", halign="center", valign="top")
                _td.add(img(style="width:%dpx" % width, src=src_file))                          # insert image into td
                #_td.add(video(style="width:%dpx" % width, src=src_file))                          # insert image into td
                _tr.add(_td)                                                                    # insert image into td

            index_so_far = vid_index + 1

        # fillup the remaining cells with blanks
        for _ in range(index_so_far, num_videos):
            for vis_type in vis_types:
                if vis_type == "reconstruction":
                    width = 4 * WIDTH
                elif vis_type == "novel view synthesis":
                    width = 3 * WIDTH
                elif vis_type == "cameras":
                    width = 3 * WIDTH
                _td = td(style="word-wrap: break-word;", halign="center", valign="top")
                _td.add(img(style="width:%dpx" % width))                                    # insert blank image into td
                _tr.add(_td)

        # add the row to the table
        t_main.add(_tr)
    
    # 8. render the table "t_main" into 
    with open("./{}.html".format(args.pagename), "w") as f: f.write(t_main.render())
