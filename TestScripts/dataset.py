### python lib
import os, sys, math, random, glob, cv2, argparse, natsort
import numpy as np

### torch lib
import torch
import torch.utils.data as data
import torchvision.transforms as tv
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

def read_img(filename, grayscale=0):

    ## read image and convert to RGB in [0, 1]

    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename)
        #img = cv2.resize(img, (0,0), fx=0.25, fy=0.25) 

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        #img = img[:, :, ::-1] ## BGR to RGB
    
    img = np.float32(img) / 255.0

    return img

crop_thingies = (0, 0, 0, 0)

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size
        print("*"*100, ih, self.ch, ih - self.ch - 0)
        self.h1 = random.randint(0, ih - self.ch - 0)
        self.w1 = random.randint(0, iw - self.cw - 0)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        global crop_thingies
        crop_thingies = (self.h1, self.h2, self.w1, self.w2)
        
    def __call__(self, img):
        #print("Came in to cropper...", crop_thingies)
        #sys.exit()
        if len(img.shape) == 3:
            return img[self.h1 : self.h2, self.w1 : self.w2, :]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]


class FixedCrop(object):
    def __init__(self, crop_size, mh, mw):
        self.ch, self.cw = crop_size
        self.h1 = mh
        self.w1 = mw
        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        #global crop_thingies
        #crop_thingies = (self.h1, self.h2, self.w1, self.w2)
        
    def __call__(self, img):
        #print("Came in to cropper...", crop_thingies)
        #sys.exit()
        if len(img.shape) == 3:
            return img[self.h1 : self.h2, self.w1 : self.w2, :]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]



class MultiFramesDataset(data.Dataset):

    def __init__(self, mode= 'train', ip_dir= './data/',
    sample_frames= 15, 
    geometry_aug= 0, scale_min= 0.5, scale_max= 2.0, crop_size= 160, order_aug= 0, size_multiplier = 2 ** 2):
        super(MultiFramesDataset, self).__init__()
        self.ip_dir = ip_dir
        self.sample_frames = sample_frames
        self.geometry_aug = geometry_aug
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.crop_size = crop_size
        self.order_aug = order_aug
        self.size_multiplier = size_multiplier
        self.mode = mode
        self.dataset_task_list = []
        self.num_frames = []
        if self.mode == "train":
            self.dataset_task_list = natsort.natsorted(glob.glob(os.path.join(self.ip_dir + "*/")))
            #list_lbl = natsort.natsorted(glob.glob(os.path.join(self.lbl_dir + "*/")))
            
            #for i in range(len(list_ip)):
            #    self.dataset_task_list.append([list_ip[i]])
        else:
            raise ValueError("Only train mode is implemented at the moment...")
        
        print("+"*20, "Total task_list:", "+"*20)
        video_names = []
        for task in self.dataset_task_list:
            video_names.append(task.split("/")[-2])
        print(video_names)
        print("+"*50)
        self.num_tasks = len(self.dataset_task_list)

        for ip in self.dataset_task_list:
            self.num_frames.append(len(natsort.natsorted(os.listdir(ip))))
        global crop_thingies

        print("[%s] Total %d videos (%d frames)" %(self.__class__.__name__, len(self.dataset_task_list), sum(self.num_frames)))


    def __len__(self):
        return len(self.dataset_task_list)


    def __getitem__(self, index):
        meta_data = {}
        meta_data['idx'] = index
        ## random select starting frame index t between [0, N - number_of_sample_frames] for "mode = "train" | "validate""
        if self.mode == "train":
            N = self.num_frames[index]
            T = random.randint(0, N - self.sample_frames)
            meta_data['starting_frame'] = T
    
            input_dir = self.dataset_task_list[index]
            #lbl_dir = self.dataset_task_list[index][1]
            meta_data['unstable_video_path'] = input_dir
            #meta_data['stable_video_path'] = lbl_dir

            ## sample from T to T + #sample_frames - 1
            frame_ip = []
            #frame_lbl = []
            ip_frame_list = natsort.natsorted(glob.glob(os.path.join(input_dir, "*.*")))
            #lbl_frame_list = natsort.natsorted(glob.glob(os.path.join(lbl_dir, "*.*")))
            for t in range(T, T + self.sample_frames):
                frame_ip.append(read_img(ip_frame_list[t]))
                #frame_lbl.append(read_img(lbl_frame_list[t]))
            meta_data['ip_frame_paths'] = ip_frame_list
            #meta_data['op_frame_paths'] = lbl_frame_list

        ## data augmentation
        if self.mode == 'train':
            
            if self.geometry_aug:

                ## random scale
                H_in = frame_ip[0].shape[0]
                W_in = frame_ip[0].shape[1]

                sc = np.random.uniform(self.scale_min, self.scale_max)
                H_out = int(math.floor(H_in * sc))
                W_out = int(math.floor(W_in * sc))

                ## scaled size should be equal to opts.crop_size
                if H_out < W_out:
                    if H_out < self.crop_size:
                        H_out = self.crop_size
                        W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
                else: ## W_out < H_out
                    if W_out < self.crop_size:
                        W_out = self.crop_size
                        H_out = int(math.floor(H_in * float(W_out) / float(W_in)))

                for t in range(self.sample_frames):
                    frame_ip[t] = cv2.resize(frame_ip[t], (W_out, H_out))
                    #frame_lbl[t] = cv2.resize(frame_lbl[t], (W_out, H_out))
                meta_data['scale_factor'] = sc

            ## random crop
            cropper = RandomCrop(frame_ip[0].shape[:2], (self.crop_size, self.crop_size))
            

            for t in range(self.sample_frames):
                frame_ip[t] = cropper(frame_ip[t])
            meta_data['crop_coords'] = crop_thingies
            

            if self.geometry_aug:
                meta_data['rotation'] = False
                ### random rotate
                rotate = random.randint(0, 3)
                if rotate != 0:
                    for t in range(self.sample_frames):
                        frame_ip[t] = np.rot90(frame_ip[t], rotate)
                        #frame_lbl[t] = np.rot90(frame_lbl[t], rotate)
                    meta_data['rotation'] = True
                    
                ## horizontal flip
                if np.random.random() >= 0.5:
                    for t in range(self.sample_frames):
                        frame_ip[t] = cv2.flip(frame_ip[t], flipCode=0)
                        #frame_lbl[t] = cv2.flip(frame_lbl[t], flipCode=0)
                    meta_data['hflip'] = True


            if self.order_aug:
                ## reverse temporal order
                meta_data['order'] = "normal"
                if np.random.random() >= 0.5:
                    meta_data['order'] = "reversed"
                    frame_ip.reverse()
                    #frame_lbl.reverse()

        ### convert (H, W, C) array to (C, H, W) tensor
        X = []
        Y = []
        for t in range(len(frame_ip)):
            X.append(torch.from_numpy(frame_ip[t].transpose(2, 0, 1).astype(np.float32)))
            #Y.append(torch.from_numpy(frame_lbl[t].transpose(2, 0, 1).astype(np.float32)))
        return {'X': X, 'meta_data': meta_data}

class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def create_data_loader(data_set, mode, train_epoch_size= 1000, batch_size= 4, threads= 8):

    ### generate random index
    if mode == 'train':
        total_samples = train_epoch_size * batch_size
    else:
        raise ValueError("Only train mode is implemented at the moment...")

    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    ### generate data sampler and loader
    sampler = SubsetSequentialSampler(indices)
    if mode == "train":
        data_loader = DataLoader(dataset= data_set, num_workers= threads, batch_size= batch_size, sampler= sampler, pin_memory= True)

    return data_loader

if __name__ == '__main__':
    train_dataset = MultiFramesDataset(sample_frames= 15, crop_size= 300)
    data_loader = create_data_loader(train_dataset, mode= "train", batch_size= 1, threads= 0)
    sane_path = "./sanitycheck/"

    for iteration, (data) in enumerate(data_loader, 1):
        x = data['X']
        if iteration > 2:
            break
        if iteration == 1:
            if not os.path.exists(sane_path):
                os.mkdir(sane_path)
            i = 0
            for xs in x:
                img = xs[0]
                img = np.transpose(img.numpy(), (1, 2, 0))
                cv2.imwrite(sane_path + "X-" + str(iteration) + "-" + str(i) + ".png", img * 255)
                i += 1

        