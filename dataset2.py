import os
import numpy as np
import cv2
import pandas as pd
#import hw1.utlis
import utlis2
import matplotlib.pyplot as plt
import torch
VIDEO_PATH="videos/"
DATASET_PATH="../HW1_dataset/"
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, dataframe):
        self.x = torch.tensor(dataframe["image"])
        #self.y = torch.tensor(dataframe["boxes"])
        self.y = dataframe["boxes"]
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        x = self.x.iloc[idx]
        y = self.y.iloc[idx]
        return x , y 


class ImageDataset(Dataset):

    def __init__(self, filename):
        self.images_names=get_images_names(filename)
        #self.x = torch.tensor(dataframe["image"])
        #self.y = torch.tensor(dataframe["boxes"])
        #self.y = dataframe["boxes"]
    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        x , y=get_image_and_box(self.images_names[idx])
    
        return x , y 



def get_images_names(name):
    """
    return a list of all the flie names in a given name (train or valid or test) 
    """
    with open(DATASET_PATH+name+".txt", "r") as f: #get classes
        text=f.readlines()
        return[t[:-5] for t in text ]
    return []

def get_images_paths(path,name):
    """
    return a list of all the  flie paths in a given name (train or valid or test) 
    """
    with open(DATASET_PATH+name+".txt", "r") as f: #get classes
        text=f.readlines()
        return[path+t for t in text ]
    return []

def get_file_labels(filename):
    """
    return a list of all the image objects: [label_index, xcenter, ycenter, w, h] in a given file 
    """
    with open("{0}bboxes_labels/{1}.txt".format(DATASET_PATH,filename), "r") as f: #get classes
        text=f.readlines()
        return[[int(v) if len(v)==1 else np.double(v) for v in t[:-2].split() ] for t in text ]#get rid of \n
    return []

def get_image_and_box(filename):
    """
    gets image name and return the image as tensor and boxes as tensor 
    """
    im=torch.tensor(cv2.imread("{0}images/{1}.jpg".format(DATASET_PATH,filename)))
    h,w,c=im.shape
    im=im.reshape([1,c,h,w]).float()
    pred=torch.tensor(get_file_labels(filename))
    return im,pred
    
    
def create_df(filename,with_image_names=False):
    """
    return a pandas df with ["ImgName","image","boxes"] from train or valid or test with image boxes and img values 
    where ImgName is the name of the image and is optional 
    """
    #df=pd.DataFrame(columns =['image',"objects"])
    if os.path.exists(filename+".pkl"):
        df = pd.read_pickle(filename+".pkl")
        return df
    data=[]
    names=get_images_names(filename)
    for name in names:
        im=cv2.imread("{0}images/{1}.jpg".format(DATASET_PATH,name))
        pred=get_file_labels(name)
        data.append([im,pred])
    df=pd.DataFrame(data,columns =['image',"boxes"])
    if with_image_names:
        df["ImgName"]=names
    df.to_pickle(filename+".pkl")
    return df

        
        
def get_dataLoader(name):
    return CustomDataset(create_df(name))

def get_dataLoader2(name):
    return ImageDataset(name)

    
def create_labels():
    labels=[]
    with open(DATASET_PATH+"classes.names", "r") as f: #get classes
        text=f.readlines()
        labels+=[t[:-2] for t in text ]#get rid of \n
    return labels

def create_db(name):
    
    full_paths=get_images_paths("HW1_dataset/images/",name)
    new_path="HW1_dataset/images/"+name+"/"
    p=new_path.replace("images","labels")
    os.mkdir(new_path)
    os.mkdir(p)
    # with open("HW1_dataset/train2.txt","w") as f:
    #     f.writelines(full_paths)
    for path in full_paths:
        path= path[:-1]
        im=cv2.imread(path)
        name=path.split("/")[-1]
        cv2.imwrite(new_path+name,im) # write im
        name=name[:-4]

        with open("{0}bboxes_labels/{1}.txt".format(DATASET_PATH,name), "r") as f: #get classes
            text=f.readlines()
            with open(p+name+".txt","w") as f:
                f.writelines(text)
            
    

def main():
    create_db("train")
    create_db("valid")
    create_db("test")
    # df=create_df("train")
    # labels=create_labels()
    # df["im"]=df.apply(lambda x: utlis.add_bboxes_to_img(x["image"],x["boxes"]),axis=1)
    
    # df["bboxes"]=df.apply(lambda x: utlis.create_boxes(x["image"].shape[:2],x["boxes"]),axis=1)
    # df["img"]=df.apply(lambda x: utlis.add_bboxes_with_labels_to_img(x["image"],x["bboxes"],labels),axis=1)
    # plt.imshow(df["im"][0])



if __name__ == "__main__":
    main()