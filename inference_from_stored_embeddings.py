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


def convertTuple(tup):
    str =  '_'.join(tup)
    return str

file_name = "Recognitions_arcface100"
network='r100'
# model = "ms1mv3_arcface_r50_fp16"
# model = "ms1mv3_arcface_r18_fp16"
model = "ms1mv3_arcface_r100_fp16"
# model = "glint360k_cosface_r50_fp16_0.1"
# model = "glint360k_cosface_r18_fp16_0.1"
# model = "glint360k_cosface_r100_fp16_0.1"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights='../weights/'+model+'/backbone.pth'
resnet = get_model(network, dropout=0)
resnet.load_state_dict(torch.load(weights, map_location=device))
resnet.eval().to(device)
print(device)

f = open(file_name+".txt", "a")
f.write("Model: "+model)
f.close()

##################################################################################################



# Searching embeddings from the folders and storing it in a list / Loading Registration embeddings

print("Registered images")

embeddings = []
names = [] 

directory = '/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/new_registered_arcface100/'
for entry in os.scandir(directory):
    folder_flag = os.path.isdir(entry)
    if (folder_flag):
        name = entry.path.split('/')[-1]
        for emb in os.scandir(entry.path):
            if (emb.path.endswith(".npy")) and emb.is_file():
                embedding_file_path = emb.path
                embedding_file_name = emb.path.split('/')[-1]
                embedding = np.load(embedding_file_path)
                embeddings.append(embedding)
                names.append(name)
                names.append(name)
                names.append(name)
                names.append(name)
                print(name)
    else:
        print("No Folders avaliable")



##################################################################################################


# Loading Attendance embeddings


attendance_images = glob.glob('/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/aligned_attendance/*')

full_path1 = []
name_list1 = []

# Storing Attendance images name in a list

for p in attendance_images:
    full = p
    str1 = p.split('/')[-1].replace(' ', '').replace('right', '').replace('random', '').replace('left', '').replace('front', '').replace('.jpg', '')
    str2 = str1.split('_')[:3]
    str3 = convertTuple(str2)
    full_path1.append(full)
    name_list1.append(str3)

# Storing Attendance embeddings in a list

attendance_names = []
attendance_embeddings = []
attendance_paths = []

print("Attendance images")

for elem in enumerate(full_path1):
    open_cv_image = cv2.imread(elem[1])
    img = open_cv_image
    img = cv2.resize(img, (112,112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    x_aligned = img
    embedding = resnet(x_aligned.to(device)) 
    attendance_embeddings.append(embedding.detach().cpu().numpy())
    attendance_paths.append(elem[1])
    # attendance_embeddings = np.vstack(attendance_embeddings)
    attendance_names.append(name_list1[elem[0]])
    print(elem[0])


##################################################################################################


# Comparing Names, Calculating the Distances and making a log file

same_count = 0
not_same_count = 0


for index1, e1 in enumerate(attendance_embeddings):
    
    distance = []

    for index2, e2 in enumerate(embeddings):

        for index3, e3 in enumerate(e2):
            
            dists = np.linalg.norm(e1 - e3)
            distance.append(dists)

    min_dist = min(distance)
    index4 = distance.index(min_dist)

    if attendance_names[index1] == names[index4]:
        same_count = same_count + 1
    else:
        not_same_count = not_same_count + 1
        
        print("\nOrignal Name: "+attendance_names[index1])
        print("Predicted Name: "+names[index4])
        print("Path: "+attendance_paths[index1])
        print("Score: "+str(min_dist))
           
        f = open(file_name+".txt", "a")
        f.write("\n\nPath: "+attendance_paths[index1])
        f.write("\nOrignal Name: "+attendance_names[index1])
        f.write("\nPredicted Name: "+names[index4])
        f.write("\nScore: "+str(min_dist))
        f.close()
        cv2.imwrite("/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/miss_pics/OrignalName_"+attendance_names[index1]+"_PredictedName_"+names[index4]+"_"+str(not_same_count)+".png",cv2.imread(attendance_paths[index1]))


print("\n\ntotal correct recognize: "+str(same_count))
print("\ntotal incorrect recognize: "+str(not_same_count))

f = open(file_name+".txt", "a")
f.write("\n\nModel: "+model)
f.write("\n\ntotal correct recognize: "+str(same_count))
f.write("\ntotal incorrect recognize: "+str(not_same_count))
f.write("\ntotal images: "+str(not_same_count+same_count))
f.close()


