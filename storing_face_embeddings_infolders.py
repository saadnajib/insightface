# from backbones import get_model
# import torch.utils.data
# import torch
# from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets, transforms
# import torchvision

# import numpy as np
# import pandas as pd
# import os
# import cv2
# import glob
# from PIL import Image, ImageDraw, ImageFont
# class CustomDataSet(Dataset):
#     def __init__(self, main_dir, transform):
#         self.main_dir = main_dir
#         self.transform = transform
#         all_folders = os.listdir(main_dir)
#         self.total_folders = all_folders

#     def __len__(self):
#         return len(self.total_folders)

#     def __getitem__(self, idx):
#         folder_loc = os.path.join(self.main_dir, self.total_folders[idx])
#         imgs = []
#         for item in os.listdir(folder_loc):
#             img_loc = os.path.join(folder_loc, item)    
#             if '.npy' in img_loc:
#                 continue
#             image = Image.open(img_loc).convert("RGB")
#             # tensor_image = self.transform(image)
#             imgs.append(image)
#         return imgs, self.total_folders[idx]

# def collate_fn(x):
#     return [y for y in x]

# data_path = '/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/gadoon_factory/folders_cosface100'
# my_dataset = CustomDataSet(data_path, transform=torchvision.transforms.ToTensor())
# train_loader = DataLoader(my_dataset , batch_size=8, shuffle=False, collate_fn=collate_fn,
#                                num_workers=1, drop_last=False)

# network='r100'
# # model = "ms1mv3_arcface_r50_fp16"
# # model = "ms1mv3_arcface_r18_fp16"
# # model = "ms1mv3_arcface_r100_fp16"
# # model = "glint360k_cosface_r50_fp16_0.1"
# # model = "glint360k_cosface_r18_fp16_0.1"
# model = "glint360k_cosface_r100_fp16_0.1"

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# weights='../weights/'+model+'/backbone.pth'
# resnet = get_model(network, dropout=0)
# resnet.load_state_dict(torch.load(weights, map_location=device))
# resnet.eval().to(device)
# print(device)

# for idx, data in enumerate(train_loader):
#     for imgs,name in data:
#         print(len(imgs), name)
#         embeddings=[]
#         for img in imgs: #iterate over images of one folder
#             img = np.array(img) #RGB image
#             img = cv2.resize(img, (112,112))
#             # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = np.transpose(img, (2, 0, 1))
#             img = torch.from_numpy(img).unsqueeze(0).float()
#             img.div_(255).sub_(0.5).div_(0.5)
#             embedding = resnet(img.to(device)) 
#             embeddings.append(embedding.detach().cpu().numpy())
#         embeddings = np.vstack(embeddings)
#         path = os.path.join(data_path,name,network+"_embeddings.npy")
#         np.save(path, embeddings)
#         print(path)
        
# # align registration faces 
# # generate tensors -> dump into file 











from backbones import get_model
import torch.utils.data
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision

import numpy as np
import pandas as pd
import os
import cv2
import glob
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import Normalizer

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_folders = os.listdir(main_dir)
        self.total_folders = all_folders

    def __len__(self):
        return len(self.total_folders)

    def __getitem__(self, idx):
        folder_loc = os.path.join(self.main_dir, self.total_folders[idx])
        imgs = []
        for item in os.listdir(folder_loc):
            img_loc = os.path.join(folder_loc, item)    
            if '.npy' in img_loc:
                continue
            image = Image.open(img_loc).convert("RGB")
            # tensor_image = self.transform(image)
            imgs.append(image)
        return imgs, self.total_folders[idx]

def collate_fn(x):
    return [y for y in x]

data_path = '/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/gadoon_factory/folders_cosface100_l2'
my_dataset = CustomDataSet(data_path, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(my_dataset , batch_size=8, shuffle=False, collate_fn=collate_fn,
                               num_workers=1, drop_last=False)

network='r100'
# model = "ms1mv3_arcface_r50_fp16"
# model = "ms1mv3_arcface_r18_fp16"
# model = "ms1mv3_arcface_r100_fp16"
# model = "glint360k_cosface_r50_fp16_0.1"
# model = "glint360k_cosface_r18_fp16_0.1"
model = "glint360k_cosface_r100_fp16_0.1"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights='../weights/'+model+'/backbone.pth'
resnet = get_model(network, dropout=0)
resnet.load_state_dict(torch.load(weights, map_location=device))
resnet.eval().to(device)
print(device)

for idx, data in enumerate(train_loader):
    for imgs,name in data:
        print(len(imgs), name)
        embeddings=[]
        for img in imgs: #iterate over images of one folder
            img = np.array(img) #RGB image
            img = cv2.resize(img, (112,112))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float()
            img.div_(255).sub_(0.5).div_(0.5)
            embedding = resnet(img.to(device))

            in_encoder = Normalizer(norm='l2')
            embedding = in_encoder.transform(embedding.detach().cpu().numpy()) 
            
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        path = os.path.join(data_path,name,network+"_embeddings.npy")
        np.save(path, embeddings)
        print(path)
        
# align registration faces 
# generate tensors -> dump into file 