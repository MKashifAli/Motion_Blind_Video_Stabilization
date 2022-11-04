import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np 
import glob
import natsort
from tqdm import tqdm
import torch
import time
from collections import OrderedDict

from dataset_test import DataGenTest
from torch.utils.data import DataLoader
from model import Model



BATCH_SIZE_VAL = 5


def path_read(unstable_path):
    loi_us = natsort.natsorted(glob.glob(os.path.join(unstable_path + "*.png")))
    scene_ds = []
    for i in range(2, len(loi_us) - 2):
        tmp = []
        tmp.append(loi_us[i - 2])
        tmp.append(loi_us[i - 1])
        tmp.append(loi_us[i])
        tmp.append(loi_us[i + 1])
        tmp.append(loi_us[i + 2])
        scene_ds.append(tmp)
    return np.asarray(scene_ds)


def load_ckp(checkpoint_name, model):
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['state_dict'])
    return model 


def load_ckp_two(checkpoint_name, model):
    state_dict = torch.load(checkpoint_name, map_location= device)['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def load_ckp_three(checkpoint_name, model):
    model.load_state_dict(torch.load(checkpoint_name))
    return model


def test(folder_to_test, checkpoint_name, model):
    val_list = path_read(folder_to_test)
    save_path = "./output_" + checkpoint_name.split(".")[0] + "/" + folder_to_test.split("/")[-2] + "/"
    print("Save path:", save_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    vds = DataGenTest(np.asarray(val_list[:, 0:5]))
    val_dl = DataLoader(vds, BATCH_SIZE_VAL, False, num_workers= 2)
    batches = int((len(vds)/BATCH_SIZE_VAL))
    loss_list = []
    pbar = tqdm(range(len(val_dl)), desc= "", ncols= 100)
    print("\nTesting scene:", folder_to_test.split("/")[-1].split(".")[0], "\n")
    with torch.no_grad():
        for (batch, (i, (x))) in zip(pbar, enumerate(val_dl)):
            tik = time.time()
            x_dev = x.to(device)
            output = model(x_dev)
            pbar.set_description(str(batch + 1) + "/" + str(batches))
            op = output.cpu().numpy()
            nop = op.shape[0]
            tok = time.time()
            save_tik = time.time()
            for pic in range(nop):
                cv2.imwrite(save_path + str(batch + 1) + "-" + str(pic) + ".png", np.transpose(op[pic], (1, 2, 0)) * 255)
 

if __name__ == "__main__":
    
    root = "./folder_containing_videos_split_into_frmaes/"
    list_of_folders = natsort.natsorted(os.listdir(root))
    
    checkpoint_name = "stage_3.pth"
    #### stg1 -> load_ckp 
    #### stg2 -> load_ckp_two
    #### stg3 -> load_ckp_three

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    model = Model(in_channels=15, out_channels=3, residual_blocks=64).to(device)
    model = load_ckp_three(checkpoint_name, model)
    model.eval()

    for folder in list_of_folders:
        folder_to_test = root + folder + "/"
        test(folder_to_test, checkpoint_name, model)