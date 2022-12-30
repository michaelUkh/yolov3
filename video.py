import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm
import predict
import bbox_visualizer as bbv
import cv2
import utlis2 as us
import dataset2 as ds
import detect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LABELS=ds.create_labels()
from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, NCOLS, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync





def tools_in_hands(pred):
    """"gets bboxes and returns list with a number for the the tools in each hand where the first entery is for right
     -1 -no hand,0-scissors ,1-needle_driver,2-forceps ,3 -no tool in hand
  """
    tools=[-1,-1]
    index=0
    for bbox in pred:
        if  bbox[0]%2==0:
            index=0 
        else:
            index=1
        if tools[index] == -1 or (tools[index] != -1 and pred[0][-1]>bbox[-1]):#
            tools[index]=bbox[0]//2
            
    return tools

def index_hand(pred,handleft=False):
    """"return the index of left or right hand in bboxs"""
    c=0
    index=-1
    
    hand= 1 if handleft else 0

    for bbox in pred:
        if  bbox[0]%2==hand:
            index=c 
            
        if index != -1 and pred[0][-1]>bbox[-1]:#
            index=1
    return index                    
    
        
    

def smooth(last_state,states,w=5):
    
    if last_state ==states[0]:
        return last_state
    else:
        tmp=[last_state]*w+states 
        return max(set(tmp), key = tmp.count)


def video_smoothing(state_right,state_left,tools_right,tools_left):
    pass  






def run(opt):
    path_video=opt.video
    look_ahead=opt.look_ahead
    check_requirements(exclude=('tensorboard', 'thop'))
    device=select_device(opt.device,printms=opt.print)

   
    
    
    
    
    cap = cv2.VideoCapture(path_video)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    i=0
    preds= []
    frames=[]
    tools_left=[]
    tools_right=[]
    state_left= 3#start with left empty hand
    state_right= 3#start with right empty hand
    left_hand=[[state_left,0]]
    right_hand=[[state_left,0]]
    video=[]
    while (cap.isOpened()):
        i += 1
        # Capture frame-by-frame
        ret, im = cap.read()
        if ret == True:
            frames.append(im)
            # add bounding boxes
            # bbox = [xmin, ymin, xmax, ymax]
            new_bboxs = predict_func(im)
            preds.append(new_bboxs)
            right, left = tools_in_hands(new_bboxs)
            tools_left.append(right)
            tools_right.append(left)
            
            if len(preds) >look_ahead:
                new_bboxs=[]
                new_left=smooth(state_left,tools_left)
                new_right=smooth(state_right,tools_right)
              
                bboxes=preds.pop(0)
                index_left,index_right=index_hand(bboxes), index_hand(bboxes,True)                
                if index_left !=-1:
                    bboxes[index_left][0]=2*new_left+1
                    new_bboxs.append(bboxes[index_left])
                if index_right !=-1:
                    bboxes[index_right][0]=2*new_right
                    new_bboxs.append(bboxes[index_right])
                conf  =[box[-1] for box in new_bboxs]
                new_bboxs=[box[:-1] for box in new_bboxs]
                
                if state_left !=new_left:
                    left_hand.append([new_left,i-look_ahead])
                    state_left= new_left
                else:
                    left_hand[-1][-1]=i-look_ahead
                
                if state_right !=new_right:
                    right_hand.append([new_right,i-look_ahead])
                    state_right= new_right
                else:
                    right_hand[-1][-1]=i-look_ahead
                    
                
                frame=frames.pop(0)
                tools_left.pop(0)
                tools_right.pop(0)
                im=us.draw_bboxes(frame,new_bboxs,LABELS)


            # Display the resulting frame
                #cv2.imshow('Frame', im)
                video.append(new_bboxs)
            print(i)

            # Press Q on keyboard to  exit
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
    video=video+preds 
    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
        
        
    
    
    if opt.save_txt:
            with open(opt.save_txt+".txt","w") as f:
                for preds in video:
                    f.write("#")
                    f.writelines([" ".join([str(n) for n in  line]) for line in preds])







def get_box_from_frame(path,name,i):
    """
    return a list of all the image objects: [label_index, xcenter, ycenter, w, h,conf] in a given file 
    """
    with open("{0}/exp/labels/{1}_{2}.txt".format(path,name,i), "r") as f: #get classes
        text=f.readlines()
        return[[int(v) if len(v)==1 else np.double(v) for v in t[:-2].split() ] for t in text ]#get rid of \n
    return []






def write_to_vid(read_dir,path_video,look_ahead,write_txt,header="/exp/labels"):
    
    name=path_video.split("/")[-1][:-4]
    cap = cv2.VideoCapture(path_video)


    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    length = len(os.listdir(read_dir+header))

    i=0
    preds= []
    frames=[]
    tools_left=[]
    tools_right=[]
    state_left= 3#start with left empty hand
    state_right= 3#start with right empty hand
    left_hand=[[state_left,0]]
    right_hand=[[state_left,0]]
    video=[]
    
    
    
    while ( i+look_ahead<length):
        i += 1
        # Capture frame-by-frame

        # add bounding boxes
        # bbox = [xmin, ymin, xmax, ymax]
        new_bboxs= get_box_from_frame(read_dir,name,i)
        preds.append(new_bboxs)
        right, left = tools_in_hands(new_bboxs)
        tools_left.append(right)
        tools_right.append(left)
        
        if len(preds) >look_ahead:
            new_bboxs=[]
            new_left=smooth(state_left,tools_left)
            new_right=smooth(state_right,tools_right)
            
            bboxes=preds.pop(0)
            index_left,index_right=index_hand(bboxes), index_hand(bboxes,True)                
            if index_left !=-1:
                bboxes[index_left][0]=2*new_left+1
                new_bboxs.append(bboxes[index_left])
            if index_right !=-1:
                bboxes[index_right][0]=2*new_right
                new_bboxs.append(bboxes[index_right])
            conf  =[box[-1] for box in new_bboxs]
            new_bboxs=[box[:-1] for box in new_bboxs]
            
            if state_left !=new_left:
                left_hand.append([new_left,i-look_ahead])
                state_left= new_left
            else:
                left_hand[-1][-1]=i-look_ahead
            
            if state_right !=new_right:
                right_hand.append([new_right,i-look_ahead])
                state_right= new_right
            else:
                right_hand[-1][-1]=i-look_ahead
                
            
            tools_left.pop(0)
            tools_right.pop(0)
            #im=us.draw_bboxes(frame,new_bboxs,LABELS)


            # Display the resulting frame
                #cv2.imshow('Frame', im)
            
    video=video+preds 

    
    
    if opt.save_txt:
            with open(read_dir+"/"+name+".txt","w") as f:
                for preds in video:
                    f.write("#\n")
                    f.writelines([" ".join([str(n) for n in  line]) for line in preds])


def run(opt):
    pass
    
    
   
    
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=2, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt




def main(opt):
   
    #write_to_vid(read_dir="res",path_video=opt.video,look_ahead=15,write_txt=False)
    detect.run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)