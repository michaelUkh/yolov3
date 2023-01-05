import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox



from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout



def fix_im(im0):
    
    img = letterbox(im0, 640, stride=32, auto=False)[0]

        # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img



def load_model(weights=ROOT / 'yolov3.pt',half=False,  # use FP16 half-precision inference
        dnn=False,device='',printms=False):
    
    
    device = select_device(device,printms=printms)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    return model



def create_predict(model,device='',conf_thres=0.25,iou_thres=0.45,classes=None, agnostic_nms=False,max_det=2,save_conf=True,half=False):
    
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx   
    device = select_device(device)

    half &= pt and device.type != 'cpu'
    
    def predict2(imr):
         # half precision only supported by PyTorch on CUDA
        
        im= fix_im(imr)
        im0s=imr
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        #dt[0] += t2 - t1

        # Inference
        visualize =  False
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        #dt[1] += t3 - t2
        preds = {}
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        #dt[2] += time_sync() - t3
        
        im0=  im0s.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        lines= []

        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = [int(cls.item()), *xywh, conf.item()] if save_conf else [int(cls.item()), *xywh]  # label format
                    lines.append(line)

                preds[i]=lines
        return lines
    return predict2


def parse_opt():
    #/ todo: fix prase_opt for us
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--im', type=str, default="../HW1_dataset/images/P016_balloon1_9.jpg", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=2, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true',default="res", help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',default=False, help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--print', action='store_true', default=False, help='print logs')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    if opt.print: 
        print_args(FILE.stem, opt)
    return opt




def run(opt):
    path_img=opt.im
    im=cv2.imread(path_img)
    check_requirements(exclude=('tensorboard', 'thop'))
    device=select_device(opt.device,printms=opt.print)
    model = load_model(weights=opt.weights,dnn=opt.dnn, device=device, printms=opt.print)
    predict_func = create_predict(model,device=device,conf_thres=opt.conf_thres,iou_thres=opt.iou_thres , 
                                  max_det=opt.max_det,classes=opt.classes,agnostic_nms=opt.agnostic_nms)
    preds = predict_func(im)
    if opt.save_txt:
          with open(opt.save_txt+".txt","w") as f:
              f.writelines([" ".join([str(n) for n in  line]) for line in preds])
    return preds

import contextlib


def main(opt):
    x=run(opt)
    print(x)
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
