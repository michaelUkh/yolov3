import numpy as np
import bbox_visualizer as bbv

def add_bboxes_to_img(im,boxes):
    """
    gets image and boxes( in the format [label_index, xcenter, ycenter, w, h] )
    and draws the boxes in the image
    Args:
        im (_type_): _description_
        boxes (_type_): _description_

    Returns:
        _type_: _description_
    """
    y,x,_ =im.shape
    boxes=create_boxes((y,x),boxes)
    bboxes=list(map(lambda x: x[1:],boxes))
    im = bbv.draw_multiple_rectangles(im, bboxes,bbox_color=(255,0,0))
    return im

def create_boxes(shape,boxes):
    """
    gets shape(y,x) and boxes( in the format [label_index, xcenter, ycenter, w, h] )

    and returns  boxes [label_index, xmin, ymin, xmax, ymax]
    """
    bboxes=[]
    y,x =shape
    for label_index, xcenter, ycenter, w, h in boxes:
        w=int(w*x)
        h=int(h*y)
        xcenter=int(x*xcenter)
        ycenter=int(y*ycenter)
        #bbox = [xmin, ymin, xmax, ymax]
        bboxes.append([label_index,xcenter,ycenter, w, h])
    return bboxes

def add_bboxes_with_labels_to_img(im,boxes,labels_name):
    """
    gets image and boxes list of box( in the format [label_index, xmin, ymin, xmax, ymax]) and labels_names 

    and return's image with the boxes and the label_name
    """
    bboxes=list(map(lambda x: x[1:],boxes))
    labels=list(map(lambda x: labels_name[x[0]],boxes))
    im = bbv.draw_multiple_rectangles(im, bboxes,bbox_color=(255,0,0))
    im=bbv.add_multiple_labels(im, labels, bboxes,text_bg_color=(255,0,0))
    return im

def draw_bboxes(im,boxes,labels_names):
    """
    gets image and boxes list of box( in the format [label_index, xcenter, ycenter, w, h]) and labels_names 

    return's image with the boxes and the label_name
    """
    boxes=create_boxes(im.shape[:2],boxes)# pixelize boxes
    return add_bboxes_with_labels_to_img(im,boxes,labels_names)
