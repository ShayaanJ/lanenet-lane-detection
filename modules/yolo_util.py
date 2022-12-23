import cv2

# get all annotations from txt files
def getAnnotations(txt):
    with open(txt, "r") as f:
        data = f.read()
        data = data.split("\n")
    return data

# removing non car/truck detections
def removeNoncar(annotation):
    for line in annotation:
        if line == "":
            annotation.remove(line)
            continue
        class_id = line.split(" ")[0]
        if int(class_id) not in [2, 7]:
            annotation.remove(line)
    return annotation

# get bottom mid coordinates of bbox of each annotation
def itsBboxTime(data):
    data = data.split(" ")
    ctrX = float(data[1]) * 1280
    ctrY = float(data[2]) * 720
    w = float(data[3]) * 1280
    h = float(data[4]) * 720

    bottomY = int(ctrY + h / 2)
    return int(ctrX), bottomY 

# overlay bottom_mid_pt on image
def overlay(img, pts):
    for i in pts:
        img = cv2.circle(img, i, 7, (212, 175, 55), -1)
    return img