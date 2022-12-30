import cv2
import os
import numpy as np
import bbox_visualizer as bbv


all_files_right = os.listdir("tools_right/")
all_files_left = os.listdir("tools_left/")

tool_usage ={"no tool in hand" : "T0",
 "needle_driver": "T1",
 "forceps": "T2",
 "scissors":"T3"}

tool_usage2 = {y:x for x,y in tool_usage.items()}
dict_class_names = {}
file_name = "P026_tissue1.txt"


labels = ['Right_Scissors','Left_Scissors','Right_Needle_driver','Left_Needle_driver','Right_Forceps',
         'Left_Forceps','Right_Empty','Left_Empty']

f_right = open("tools_right/"+file_name, "r")
f_left = open("tools_left/"+file_name, "r")
r_lines = f_right.readlines()
l_lines = f_left.readlines()
r_time_tool = {}
for line in r_lines:
    data = line.split()
    for i in range(int(data[0]),int(data[1])+1):
        r_time_tool[i] = tool_usage2[data[2]]
l_time_tool = {}
for line in l_lines:
    data = line.split()
    for i in range(int(data[0]),int(data[1])+1):
        l_time_tool[i] = tool_usage2[data[2]]

cap = cv2.VideoCapture('P026_tissue1.wmv')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
i=0
success,image = cap.read()
count = 0
x,y,w,h = 0,0,400,250


img_array = []
while success:
    #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = cap.read()
    try:
        text_r = r_time_tool[count]
        text_l = l_time_tool[count]
        height, width, layers = image.shape
        size = (width, height)
        cv2.putText(img=image, text="r:" + text_r + " l:" + text_l, org=(x + int(w / 15), y + int(h / 3)),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5, color=(0, 0, 255), thickness=1)
        img_array.append(image)
        #im = draw_bboxes(image, , )

    except:
        print(success)

    count += 1
    if count % 1000 == 0:
        print(count)
out = cv2.VideoWriter('P026_tissue1_new.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
