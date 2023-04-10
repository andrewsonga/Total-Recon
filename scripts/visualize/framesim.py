"""
bash scripts/compute_framesim.sh
"""
from absl import flags, app
import sys
sys.path.insert(0,'')
sys.path.insert(0,'third_party')
import numpy as np
import torch
import os
import glob
import imageio
from nnutils.train_utils import v2s_trainer
import matplotlib.pyplot as plt
opts = flags.FLAGS

def main(_):
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)

    model = trainer.model
    model.eval()

    env_embeddings = model.env_code.weight
    frame_ids = torch.arange(end = model.pose_code.vid_offset[-1]).to(model.device)
    pose_embeddings = model.pose_code(frame_ids)
    
    #print("model.env_code: {}".format(model.env_code.weight.shape))     # (num_fr = 1400, embed_dim = 64)
    #print("pose_embeddings: {}".format(pose_embeddings.shape))          # (num_fr = 1400, embed_dim = 128)

    # create similarity matrix

    # for pose codes
    dot_prod_embedpose = torch.mm(pose_embeddings, pose_embeddings.transpose(1, 0))     # (num_fr, num_fr)
    embedpose_norm = torch.norm(pose_embeddings, dim = -1, keepdim=True)                # (num_fr, 1)
    cossim_embedpose = torch.div(dot_prod_embedpose, torch.mm(embedpose_norm, embedpose_norm.transpose(1, 0) + 1e-10))

    # for env codes
    dot_prod_embedenv = torch.mm(env_embeddings, env_embeddings.transpose(1, 0))     # (num_fr, num_fr)
    embedenv_norm = torch.norm(env_embeddings, dim = -1, keepdim=True)                # (num_fr, 1)
    cossim_embedenv = torch.div(dot_prod_embedenv, torch.mm(embedenv_norm, embedenv_norm.transpose(1, 0) + 1e-10))

    print(cossim_embedpose[cossim_embedpose < 0])
    print(cossim_embedenv[cossim_embedenv < 0])

    plt.imsave("./cossim_embedpose.png", cossim_embedpose.detach().cpu().numpy(), vmin=-1., vmax=1., cmap="inferno")
    plt.imsave("./cossim_embedenv.png", cossim_embedenv.detach().cpu().numpy(), vmin=-1., vmax=1., cmap="inferno")

if __name__ == '__main__':
    app.run(main)