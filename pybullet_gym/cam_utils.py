import cv2
import os
import numpy as np

def find_link_px(img_mask, goal_value, w, h):
    index = [i for i, x in enumerate(img_mask) if x == goal_value]
    x = [(i % w) for i in index]
    y = [int(i / w) for i in index]
    return [int(sum(x) / len(x)), int(sum(y) / len(y))]

def save_set(id, img_lst, center, goal_label):
    path = "./dataset/"
    name = "traj"+str(id)
    file = os.path.join(path, name)
    if not os.path.exists(file):
        os.mkdir(file)

    #save image
    for idx, img in enumerate(img_lst):
        cv2.imwrite(
            os.path.join(file, "img"+str(idx).zfill(3)+".jpg"), img)
    goal_txt = open(os.path.join(file, "goal.txt"),'w')
    goal_txt.write("{}\n".format(center))
    goal_txt.write("{}".format(goal_label))

    #write txt
    file_object = open(path+"info.txt", 'a')
    file_object.write("{},{}\n".format(name, center))

    print("save data {} successfully.".format(file))


