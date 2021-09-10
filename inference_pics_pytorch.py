from operator import truediv
from torch.types import Number
from backbones import get_model
import torch.utils.data
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
import time
import numpy as np
import pandas as pd
import os
import cv2
import glob
from PIL import Image, ImageDraw, ImageFont
# from retinaface import RetinaFace
from openvino.inference_engine import IENetwork, IECore
import os.path as osp
# import spdlog
from FaceDetection.py_cpu_nms import py_cpu_nms
from FaceDetection.post_processing_utils import PriorBox
from FaceDetection.post_processing_utils import decode
from FaceDetection.post_processing_utils import decode_landm
from FaceDetection.config_rfb import cfg_rfb
import FaceAlignment.face_alignment as face_alignment
from sklearn.preprocessing import Normalizer





##################################################################################################

CONF_THRESHOLD = 0.02
TOP_K = 1000
NMS_THRESHOLD = 0.4
VIEW_THRESHOLD = 0.9

network='r34'
# model = "ms1mv3_arcface_r50_fp16"
# model = "ms1mv3_arcface_r18_fp16"
# model = "ms1mv3_arcface_r100_fp16"
# model = "glint360k_cosface_r50_fp16_0.1_new_trained"
# model = "glint360k_cosface_r50_fp16_0.1"
model = "glint360k_cosface_r34_fp16_0.1"
# model = "glint360k_cosface_r18_fp16_0.1"
# model = "glint360k_cosface_r100_fp16_0.1"
generate_embedding = True

reg_embedding_dataset_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/testing_dataset4/output_big/registration"

landmark_flag = False
bounding_box_flag = False
test_dataset_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/testing_dataset4/output_big/test"

correct = 0
total_images = 0
incorrect = 0
incorrect_flag = False
true_positive = 0
true_negative = 0

##################################################################################################

# functions 
def embeddings_gen(network,model,reg_embedding_dataset_path,resnet,net_landms):
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

    data_path = reg_embedding_dataset_path
    my_dataset = CustomDataSet(data_path, transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(my_dataset , batch_size=8, shuffle=False, collate_fn=collate_fn,
                                num_workers=1, drop_last=False)

    print(device)

    for idx, data in enumerate(train_loader):
        for imgs,name in data:
            print(len(imgs), name)
            embeddings=[]
            for img in imgs: #iterate over images of one folder
                img = np.array(img) #RGB image
                dets = get_landmarks(img, net_landms)
                aligned_faces,bb = align_faces(dets, img)
                for i, face in enumerate(aligned_faces):

                    img = face
                    # cv2.imshow("",img)
                    # cv2.waitKey(0)

                    # img = cv2.resize(img, (112,112))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = np.transpose(img, (2, 0, 1))
                    img = torch.from_numpy(img).unsqueeze(0).float()
                    img.sub_(0.5).div_(0.5)
                    embedding = resnet(img.to(device))

                    in_encoder = Normalizer(norm='l2')
                    embedding = in_encoder.transform(embedding.detach().cpu().numpy()) 
                    
                    embeddings.append(embedding)
            embeddings = np.vstack(embeddings)
            path = os.path.join(data_path,name,network+"_embeddings.npy")
            np.save(path, embeddings)
            print(path)


def convertTuple(tup):
    str =  '_'.join(tup)
    return str


def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized

def get_landmarks(img, net):
    blob = cv2.dnn.blobFromImage(img, size=(48, 48))
    output_layer = net.getUnconnectedOutLayersNames()
    net.setInput(blob)
    landms = net.forward(output_layer)
    landms = np.array(landms)
    landms = landms.reshape(5,2)
    scale = [img.shape[1], img.shape[0]]
    landms = np.int32(landms * scale)
    # Concatenating detections and landmarks
    dets = np.concatenate((np.array([0,0,img.shape[1], img.shape[0], 1]), landms.flatten()))
    return np.expand_dims(dets, axis=0)

def get_detections(img, net):

    # Prepare input blob and perform inference
    blob = cv2.dnn.blobFromImage(img, size=(640, 480), mean=(104, 117, 123))

    net.setInput(blob)

    output_layer = net.getUnconnectedOutLayersNames()
    boxes, landms, scores = net.forward(output_layer)

    loc = np.array(boxes)

    # Post-processing the outputs (Decoding, NMS etc.)
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    resize = 1  # Not resizing for now

    priorbox = PriorBox(cfg_rfb, image_size=(480, 640))
    prior_data = priorbox.forward()

    boxes = decode(loc.squeeze(0), prior_data, cfg_rfb['variance'])
    boxes = boxes * scale / resize

    scores = np.array(scores).squeeze(0)[:, 1]
    landms = decode_landm(landms.squeeze(0), prior_data, cfg_rfb['variance'])
    scale1 = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                            img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                            img.shape[1], img.shape[0]])

    landms = landms * scale1 / resize

    # Ignoring low scores
    inds = np.where(scores > CONF_THRESHOLD)[0]

    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # Keeping top-K before NMS
    order = scores.argsort()[::-1][:TOP_K]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # Doing NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, NMS_THRESHOLD)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:TOP_K, :]
    landms = landms[:TOP_K, :]

    dets = np.concatenate((dets, landms), axis=1)

    return dets


def align_faces(dets, img):

    img = cv2.dnn.blobFromImage(img, scalefactor=1./255)

    aligned_faces = []  #np.array([], dtype=object)
    bb = [] 

    for b in dets:
        if b[4] < VIEW_THRESHOLD:
            continue
        b = list(map(int, b))
        roi = face_alignment.ROI(b)
        face = face_alignment.cut_roi(img, roi)

        lms = np.array([b[5:15]], dtype=np.float64).reshape(5,2)
        lms -= np.array([roi.position[0], roi.position[1]])
        face = np.expand_dims(face, axis=0)

        face_alignment.align_faces(face, lms.reshape(1,5,2))
        orig_face = face[0][0].transpose(1,2,0)

        # cv2.imshow(" ",orig_face)
        # cv2.waitKey(1)
        
        bb.append([roi.position.astype(int),(roi.position + roi.size).astype(int)])
        aligned_faces.append(orig_face)

    return aligned_faces,bb

##################################################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights='../weights/'+model+'/backbone.pth'
resnet = get_model(network, dropout=0)
resnet.load_state_dict(torch.load(weights, map_location=device))
resnet.eval().to(device)
net = cv2.dnn.readNet("rfb.bin", "rfb.xml")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
print(device)
net_landms = cv2.dnn.readNet("Models/landmarks-regression-retail-0009.bin", "Models/landmarks-regression-retail-0009.xml")

##################################################################################################

# Creating embedding and storing them in a file format in dataset directory

if generate_embedding == True:
    embeddings_gen(network,model,reg_embedding_dataset_path,resnet,net_landms)

##################################################################################################

# Searching embeddings from the folders and storing it in a list / Loading Registration embeddings

print("Registered images")

reg_embeddings = []
names = [] 


directory = reg_embedding_dataset_path
for entry in os.scandir(directory):
    folder_flag = os.path.isdir(entry)
    if (folder_flag):
        name = entry.path.split('/')[-1]
        for emb in os.scandir(entry.path):
            if (emb.path.endswith(".npy")) and emb.is_file():
                embedding_file_path = emb.path
                embedding_file_name = emb.path.split('/')[-1]
                embedding = np.load(embedding_file_path)
                reg_embeddings.append(embedding)
                number_of_emb = embedding.shape[0]
                for i in range(number_of_emb):
                    names.append(name)
                print(name)
    else:
        print("No Folders avaliable")



##################################################################################################
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
                imgs.append(image)
            return imgs, self.total_folders[idx]

def collate_fn(x):
        return [y for y in x]

data_path = test_dataset_path
my_dataset = CustomDataSet(data_path, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(my_dataset , batch_size=8, shuffle=False, collate_fn=collate_fn,
                            num_workers=1, drop_last=False)

# print(device)

for idx, data in enumerate(train_loader):
    for imgs,name in data:
        # print(len(imgs), name)
        for img in imgs:
            actual_name = name
     
            try:
                total_images = total_images + 1
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # cv2.imshow("attendance",img)
                frame = img
                start_time = time.time()

                plain_frame = frame.copy()

                dets = get_landmarks(frame, net_landms)
                # dets = get_detections(frame, net)
                aligned_faces,bb = align_faces(dets, frame)

                for i, face in enumerate(aligned_faces):
                    
                    img = face
                    # cv2.imshow('img', img)
                    # cv2.waitKey(0)

                    # img = cv2.resize(img, (112,112))
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img = np.transpose(img, (2, 0, 1))
                    img = torch.from_numpy(img).unsqueeze(0).float()
                    img.sub_(0.5).div_(0.5)
                    x_aligned = img
                    attendance_embedding = resnet(x_aligned.to(device))

                    in_encoder = Normalizer(norm='l2')
                    attendance_embedding = in_encoder.transform(attendance_embedding.detach().cpu().numpy())
                    attendance_embedding = attendance_embedding


                    distance = []

                    for index2, e2 in enumerate(reg_embeddings):
                        for index3, e3 in enumerate(e2):
                            dists = np.linalg.norm(attendance_embedding - e3)
                            distance.append(dists)
                    min_dist = min(distance)
                    index4 = distance.index(min_dist)

                    if min_dist <= 2:
                            Predicted_name = names[index4]
                            check_name = Predicted_name
                            
                    else:
                        Predicted_name = "unknown"
                        
                    print("Predicted name: "+Predicted_name)
                    print("score: ",min_dist)
                    print("Actual name: "+actual_name)
                    print("\n")
                    if Predicted_name == actual_name:
                        true_positive = true_positive + 1
                        
                    else:
                        for n in names:
                            if actual_name == n:
                                incorrect = incorrect + 1 #false negative
                                incorrect_flag = True
                                break

                        if incorrect_flag == False and Predicted_name == "unknown":
                            true_negative = true_negative + 1
                        else:
                            incorrect = incorrect + 1 #false positive

                    #################################### printing landmarks on frame #################################### 
                    if landmark_flag == True:
                        for b in dets:
                            if b[4] < VIEW_THRESHOLD:
                                continue
                            text = "{:.4f}".format(b[4])
                            b = list(map(int, b))

                            # landms
                            cv2.circle(frame, (b[5], b[6]), 4, (0, 0, 255), -1)
                            cv2.circle(frame, (b[7], b[8]), 4, (0, 255, 255), -1)
                            cv2.circle(frame, (b[9], b[10]), 4, (255, 0, 255), -1)
                            cv2.circle(frame, (b[11], b[12]), 4, (0, 255, 0), -1)
                            cv2.circle(frame, (b[13], b[14]), 4, (255, 0, 0), -1)

                    #################################### printing bounding boxes  #################################### 
                    
                    if bounding_box_flag == True:
                        cv2.rectangle(frame, bb[i][0],bb[i][1], (0, 0, 255), 2)
                    
                    # bb[0][0][0] = b[0]
                    # bb[0][0][1] = b[1]
                    # bb[0][1][0] = b[2]
                    # bb[0][1][1] = b[3]

                    cx = bb[i][0][0]
                    cy = bb[i][0][1] + 12
                    cy1 = bb[i][0][1] - 12

                    cv2.putText(frame, Predicted_name, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255),1)

                end_time = time.time()
            except:
                print("something went wrong")
                continue
        
######################################### printing FPS #########################################
fps = total_images / (end_time - start_time)
print("Estimated frames per second : {0}".format(fps))
################################################################################################
correct = true_negative + true_positive
percentage = (correct/total_images)*100
print("Total Correct :"+ str(correct))
print("Total True Positive :"+ str(true_positive))
print("Total True Negative :"+ str(true_negative))
print("Total Incorrect :"+ str(incorrect))
print("Total Images :"+ str(total_images))
print("Correct Recognition percentage : {0:.2f}%".format(percentage))
