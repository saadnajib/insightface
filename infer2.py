from backbones import get_model
import torch.utils.data
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
import cv2
import glob
from PIL import Image, ImageDraw, ImageFont




# Face detection and recognition inference pipeline
# The following example illustrates how to use the facenet_pytorch python package to perform face detection
# and recogition on an image dataset using an Inception Resnet V1 pretrained on the VGGFace2 dataset.
# The following Pytorch methods are included:

#     Datasets
#     Dataloaders
#     GPU/CPU processing



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
network='r18'
# model = "ms1mv3_arcface_r50_fp16"
model = "ms1mv3_arcface_r18_fp16"
# model = "glint360k_cosface_r50_fp16_0.1"
# model = "glint360k_cosface_r18_fp16_0.1"
weights='/home/saad/saad/arcface/insightface/recognition/arcface_torch/weights/'+model+'/backbone.pth'
resnet = get_model(network, dropout=0)
resnet.load_state_dict(torch.load(weights, map_location=device))
resnet.eval().to(device)



def convertTuple(tup):
    str =  '_'.join(tup)
    return str

attendance_images = glob.glob('/home/saad/saad/arcface/traning_dataset/gadoon office/test2/attendance/*')
register_images = glob.glob('/home/saad/saad/arcface/traning_dataset/gadoon office/test2/register/*')

full_path1 = []
name_list1 = []
full_path2 = []
name_list2 = []


for p in attendance_images:
    full = p
    str1 = p.split('/')[-1].replace(' ', '').replace('right', '').replace('random', '').replace('left', '').replace('front', '').replace('.jpg', '')
    str2 = str1.split('_')[:3]
    str3 = convertTuple(str2)
    full_path1.append(full)
    name_list1.append(str3)

for r in register_images:
    full = r
    str1 = r.split('/')[-1].replace(' ', '').replace('right', '').replace('random', '').replace('left', '').replace('front', '').replace('.jpg', '')
    str2 = str1.split('_')[:3]
    str3 = convertTuple(str2)
    full_path2.append(full)
    name_list2.append(str3)

register_names = []
attendance_names = []
register_embeddings = []
attendance_embeddings = []

# elem[0] = number
# elem[1] = path

for elem in enumerate(full_path1):

    open_cv_image = cv2.imread(elem[1])
    img = open_cv_image
    img = cv2.resize(img, (112,112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    x_aligned = img

    embedding = resnet(x_aligned).detach().cpu()
    attendance_embeddings.append(embedding)
    attendance_names.append(name_list1[elem[0]])
    print(elem[0])


for elem in enumerate(full_path2):

    open_cv_image = cv2.imread(elem[1])
    img = open_cv_image
    img = cv2.resize(img, (112,112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    x_aligned = img

    embedding = resnet(x_aligned).detach().cpu()
    register_embeddings.append(embedding)
    register_names.append(name_list2[elem[0]])
    print(elem[0])



# Print distance matrix for classes

same_count = 0
not_same_count = 0

for num1,e1 in enumerate(attendance_embeddings):
    dists1 = 10000000000000
    for  num2,e2 in enumerate(register_embeddings):
        # n1 = attendance_names[num1]
        # n2 = register_names[num2]
        dists = (e1 - e2).norm().item()

        if (dists <= dists1):
            dists1 = dists
            unique_num = num2

    print("Orignal Name: ", attendance_names[num1])
    print("Predicted Name: ",register_names[unique_num])
    print("Percentage: ",dists1)

    if attendance_names[num1] == register_names[unique_num]:
        same_count = same_count + 1
    else:
        not_same_count = not_same_count + 1  
        f = open("Recognitions.txt", "a")
        f.write("\n\nOrignal Name: "+attendance_names[num1])
        f.write("\nPredicted Name: "+register_names[unique_num])
        f.write("\nScore: "+str(dists1))
        f.close()


print("\n\ntotal correct recognize: "+str(same_count))
print("\ntotal incorrect recognize: "+str(not_same_count))

f = open("Recognitions.txt", "a")
f.write("\n\ntotal correct recognize: "+str(same_count))
f.write("\ntotal incorrect recognize: "+str(not_same_count))
f.close()



