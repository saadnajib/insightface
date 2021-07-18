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




##################################################################################################

video_speedup = False
CONF_THRESHOLD = 0.02
TOP_K = 1000
NMS_THRESHOLD = 0.4
VIEW_THRESHOLD = 0.9

##################################################################################################

# functions 

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

    # Showing output on the image
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
        # cv2.waitKey(0)
        
        aligned_faces.append(orig_face)

    return aligned_faces

##################################################################################################

file_name = "Recognitions_cosface100_saad"
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
net = cv2.dnn.readNet("rfb.bin", "rfb.xml")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
print(device)

f = open(file_name+".txt", "a")
f.write("Model: "+model)
f.close()

##################################################################################################


# Searching embeddings from the folders and storing it in a list / Loading Registration embeddings

print("Registered images")

reg_embeddings = []
names = [] 

directory = '/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/all_new_registered_new_cosface100/'
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
                names.append(name)
                names.append(name)
                names.append(name)
                names.append(name)
                print(name)
    else:
        print("No Folders avaliable")



##################################################################################################


# Loading Video

video_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/input_video/saad2.mp4"
save_video = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/saved_video/output3.avi"


# inital camera
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap = cv2.VideoCapture(video_path)
cap.set(3,500)
cap.set(4,500)   
_, frame = cap.read()

frame = resize_image(frame, 0.5)
out = cv2.VideoWriter(save_video,cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame.shape[1], frame.shape[0]), isColor=True)
total_frames_passed = 0


while cap.isOpened():

        
    isSuccess,frame = cap.read()
        
    if video_speedup:               
        total_frames_passed += 1
        if total_frames_passed % video_speedup != 0:
            continue

    if isSuccess:            
        try:
            ######### TO write a frame #########
            # cv2.imwrite("/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/saved_video/saad.jpg",frame)
            
            # image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            start_time = time.time()
            frame = resize_image(frame, 0.5)

            dets = get_detections(frame, net)
            aligned_faces = align_faces(dets, frame)

            for i, face in enumerate(aligned_faces):
                
                ######### TO write cropped face image #########
                # cv2.imwrite(save_path+""+name_list1[elem[0]]+"_"+str(elem[0])+"_"+str(i)+".jpg",255*face)
                
                img = 255*face
                # cv2.imshow('img', img)
                # cv2.waitKey(1)

                img = cv2.resize(img, (112,112))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
                img = torch.from_numpy(img).unsqueeze(0).float()
                img.div_(255).sub_(0.5).div_(0.5)
                x_aligned = img
                attendance_embedding = resnet(x_aligned.to(device))
                attendance_embedding = attendance_embedding.detach().cpu().numpy()

                distance = []

                for index2, e2 in enumerate(reg_embeddings):
                    for index3, e3 in enumerate(e2):
                        dists = np.linalg.norm(attendance_embedding - e3)
                        distance.append(dists)
                min_dist = min(distance)
                index4 = distance.index(min_dist)

                if min_dist <= 27:
                    Predicted_name = names[index4]
                    
                else:
                    Predicted_name = "unknown"
                    
                
                print(Predicted_name)
                print("score: ",min_dist)

                #################################### printing bounding boxes and landmarks on frame #################################### 
                for b in dets:
                    if b[4] < VIEW_THRESHOLD:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    cv2.rectangle(frame, (b[0], b[1]),(b[2], b[3]), (0, 0, 255), 2)
                    
                    cx = b[0]
                    cy = b[1] + 12

                    cv2.putText(frame, Predicted_name, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    # landms
                    cv2.circle(frame, (b[5], b[6]), 5, (0, 0, 255), -1)
                    cv2.circle(frame, (b[7], b[8]), 5, (0, 255, 255), -1)
                    cv2.circle(frame, (b[9], b[10]), 5, (255, 0, 255), -1)
                    cv2.circle(frame, (b[11], b[12]), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (b[13], b[14]), 5, (255, 0, 0), -1)


                FPS = 1.0 / (time.time() - start_time)

                frame = cv2.putText(frame,
                        'FPS: {:.1f}'.format(FPS),(10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,
                        (255,0,0),
                        2,
                        cv2.LINE_AA)
         
        except:
            continue

        out.write(frame) 
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

cap.release()
out.release()
cv2.destroyAllWindows()




