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

video_speedup = False
CONF_THRESHOLD = 0.02
TOP_K = 1000
NMS_THRESHOLD = 0.4
VIEW_THRESHOLD = 0.9
save_attendance = True
save_attendance_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/saved_video/save_attendance/new_saving_faces_cosface50_6video"
# save_attendance_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/saved_video/save_attendance/new_saving_faces_cosface50_2video_again"

file_name = "Recognitions_cosface100_saad"
network='r50'
# model = "ms1mv3_arcface_r50_fp16"
# model = "ms1mv3_arcface_r18_fp16"
# model = "ms1mv3_arcface_r100_fp16"
# model = "glint360k_cosface_r50_fp16_0.1_new_trained"
model = "glint360k_cosface_r50_fp16_0.1"
# model = "glint360k_cosface_r34_fp16_0.1"
# model = "glint360k_cosface_r18_fp16_0.1"
# model = "glint360k_cosface_r100_fp16_0.1"
generate_embedding = False
# reg_embedding_dataset_path = '/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/gadoon_factory/folders_cosface100/'
# reg_embedding_dataset_path = '/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/gadoon_factory/folders_cosface50_l2/'
# reg_embedding_dataset_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/gadoon_factory/agligned_registration/aligned_faces_folders_cosface50_newtrainedmodel"
reg_embedding_dataset_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/gadoon_factory/agligned_registration/aligned_faces_folders_cosface50"

input_video_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/gadoon_factory/videos/6.mp4"
# input_video_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/input_video/test1.mp4"
# input_video_path = "/home/saad/Videos/test_video3/7.mp4"
save_video_flag = False
save_video = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/saved_video/cosface50_1-th_trainedmodel.avi"
landmark_flag = False
bounding_box_flag = True

##################################################################################################

# functions 

def embeddings_gen(network,model,reg_embedding_dataset_path,resnet):
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

# f = open(file_name+".txt", "a")
# f.write("Model: "+model)
# f.close()

##################################################################################################

# Creating embedding and storing them in a file format in dataset directory

if generate_embedding == True:
    embeddings_gen(network,model,reg_embedding_dataset_path,resnet)


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

# Loading Video

video_path = input_video_path

# inital camera
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap = cv2.VideoCapture(video_path)
cap.set(3,500)
cap.set(4,500)  
_, frame = cap.read()

frame1 = resize_image(frame, 0.5)
if save_video_flag == True:
    out = cv2.VideoWriter(save_video,cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame1.shape[1], frame1.shape[0]), isColor=True)
total_frames_passed = 0


while cap.isOpened():

        
    isSuccess,frame = cap.read()
        
    if video_speedup:               
        total_frames_passed += 1
        if total_frames_passed % video_speedup != 0:
            continue

    if isSuccess:            
        try:
            ######### TO write every frame #########
            # cv2.imwrite("/home/saad/saad/arcface/insightface/recognition/arcface_torch/aligned_embeddings_omair_shared/saved_video/saad.jpg",frame)
            
            # image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            start_time = time.time()
            # frame = resize_image(frame, 0.5)
            plain_frame = frame.copy()

            dets = get_detections(frame, net)
            aligned_faces,bb = align_faces(dets, frame)

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

                in_encoder = Normalizer(norm='l2')
                attendance_embedding = in_encoder.transform(attendance_embedding.detach().cpu().numpy())
                attendance_embedding = attendance_embedding

                # attendance_embedding = attendance_embedding.detach().cpu().numpy()

                distance = []

                for index2, e2 in enumerate(reg_embeddings):
                    for index3, e3 in enumerate(e2):
                        dists = np.linalg.norm(attendance_embedding - e3)
                        # dists = np.dot(attendance_embedding,e3)/np.cross(np.linalg.norm(attendance_embedding), np.linalg.norm(e3))
                        distance.append(dists)
                min_dist = min(distance)
                index4 = distance.index(min_dist)

                if min_dist <= 1.0:
                    Predicted_name = names[index4]
                    check_name = Predicted_name
                    
                else:
                    Predicted_name = "unknown"
                    
                print(Predicted_name)
                print("score: ",min_dist)
                print("\n")

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
                
                if save_attendance == True:
                    plain_frame = cv2.copyMakeBorder(plain_frame, 70, 70, 70, 70, cv2.BORDER_CONSTANT)    
                    frame_face_crop = plain_frame[(bb[i][0][1]-70)+70:(bb[i][1][1]+70)+70, (bb[i][0][0]-70)+70:(bb[i][1][0]+70)+70]

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


                FPS = 1.0 / (time.time() - start_time)

                frame = cv2.putText(frame,
                        'FPS: {:.1f}'.format(FPS),(10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,
                        (255,0,0),
                        2,
                        cv2.LINE_AA)

                #################################### saving attendance ####################################


                if Predicted_name != "unknown" and save_attendance == True:
                    
                    name_list = []
                    flag_solo = False

                    if not os.path.exists(save_attendance_path+'/solo'):
                        os.makedirs(save_attendance_path+'/solo')
                        img = np.zeros((512, 512, 1), dtype = "uint8")
                        cv2.imwrite(save_attendance_path+"/solo/random_image_12345.jpg",img)
                        


                    photos = glob.glob(save_attendance_path+'/solo/*')
                    #str3 = Ghulam_Mustafa_77796
                    for p in photos:
                        str1 = p.split('/')[-1].replace(' ', '').replace('.jpg', '')
                        str2 = str1.split('_')[:3]
                        str3 = convertTuple(str2)
                        name_list.append(str3)
                    
                    for i in name_list:
                        if check_name == i: 
                            flag_solo = False
                            break
                        else:
                            flag_solo = True
    
                    if flag_solo == True:
                        cv2.imwrite(save_attendance_path+"/solo/"+Predicted_name+"_score_"+str(min_dist)+".jpg",frame_face_crop)
                        f = open(save_attendance_path+"/solo/recognitions.txt","a")
                        f.write("\nPredicted name: "+Predicted_name)
                        f.close()

                    cv2.imwrite(save_attendance_path+"/"+Predicted_name+"_score_"+str(min_dist)+".jpg",frame_face_crop)
         
        except:
            continue
        
        if save_video_flag == True:
            frame1 = resize_image(frame, 0.5) 
            out.write(frame1)
        
        frame2 = resize_image(frame, 0.5) 
        cv2.imshow('video', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        os.remove(save_attendance_path+"/solo/random_image_12345.jpg")
        break  

cap.release()

if save_video_flag == True:
    out.release()

cv2.destroyAllWindows()




