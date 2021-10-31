from ctypes import *
import math
import os
import cv2
import numpy as np
import time
import darknet

def is_close(p1, p2):
    """
    #================================================================
    # 1. Purpose : Calculate Euclidean Distance between two points
    #================================================================    
    :param:
    p1, p2 = two points for calculating Euclidean Distance
    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1**2 + p2**2)
    #=================================================================#
    return dst 

def convertBack(x, y, w, h): 
    #================================================================
    # 2.Purpose : Converts center coordinates to rectangle coordinates
    #================================================================  
    """
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img,):
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet
    :return:
    img with bbox
    """
    global count, objek
    #================================================================
    # 3.1 Purpose : Filter out Persons class from detections and get 
    #               bounding box centroid for each person detection.
    #================================================================
    if len(detections) > 0:  						# At least 1 detection in the image and check detection presence in a frame  
        centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
        objectId = 0								# We inialize a variable called ObjectId and set it to 0
        distance = 100
        for label, confidence, bbox in detections:				# In this if statement, we filter all the detections for persons only
            x, y, w, h = (bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3])
            name_tag = label      	# Store the center points of the detections
            # xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox            
            # Append center point of bbox for persons detected.
            centroid_dict[objectId] = (name_tag, confidence, int(x), int(y), int(w), int(h)) # Create dictionary of tuple with 'objectId' as the index center points and bbox
            objectId += 1
        for idx in centroid_dict.items():
            print(idx)
            if float(idx[1][1]) > 50:
                if objek[0] == []:
                    temp = (idx[1][0], idx[1][2], idx[1][3], 1)
                    objek[0] = np.append(idx[0], temp)
                else:
                    for row in objek:
                        if idx[0] == row[0] and idx[1][0] == row[1]:
                            dx = int(idx[1][2]) - int(row[2])
                            dy = int(idx[1][3]) - int(row[3])
                            if distance > is_close(dx, dy):
                                row[2] = idx[1][2]
                                row[3] = idx[1][3]
                            else:
                                row[2] = idx[1][2]
                                row[3] = idx[1][3]
                                row[4] = int(row[4]) + 1

                        elif idx[0] == row[0] and idx[1][0] != row[1]:
                            dx = int(idx[1][2]) - int(row[2])
                            dy = int(idx[1][3]) - int(row[3])
                            if distance > is_close(dx, dy):
                                row[2] = idx[1][2]
                                row[3] = idx[1][3]
                            else:
                                temp = (idx[0], idx[1][0], idx[1][2], idx[1][3], 1)
                                objek = np.vstack([objek, temp])
                        elif idx[0] != row[0]:
                            exist = False
                            for names in objek:
                                if idx[1][0] == names[1]:
                                    exist = True
                                    dx = int(idx[1][2]) - int(names[2])
                                    dy = int(idx[1][3]) - int(names[3])
                                    if distance > is_close(dx, dy):
                                        names[2] = idx[1][2]
                                        names[3] = idx[1][3]
                                    else:
                                        names[2] = idx[1][2]
                                        names[3] = idx[1][3]
                                        names[4] = int(names[4])+1
                            if exist == False:
                                temp = (idx[0], idx[1][0], idx[1][2], idx[1][3], 1)
                                objek = np.vstack([objek, temp])

                        



        #     if objek == {}:
        #         objek[count] = (name_tag, objectId, 1, x, y)
        #         count += 1
        #     objectId += 1 #Increment the index for each detection
        #     print(centroid_dict)
        # for idx, box in centroid_dict.items():
        #     for jumlah in objek.items():
        #         if box[0] == jumlah[0]:
        #             dx = int(box[2]-jumlah[2])
        #             dy = int(box[3]-jumlah[3])
        #             jarak = is_close(dx, dy)
        #             if distance > jarak:
        #                 pass

        #     if float(box[1]) > 50:
        #         cv2.rectangle(img, (box[4], box[5]), (box[6], box[7]), (255, 0, 0), 2)
    print(objek)      
    #=================================================================#
    return img

netMain = None
metaMain = None
altNames = None

count = 0
objek = [[]]

def YOLO():
    """
    Perform Object detection
    """

    global metaMain, netMain, altNames, objek
    configPath = "./cfg/custom-yolov4-tiny-detector.cfg"
    weightPath = "./custom-yolov4-tiny-detector_last.weights"
    metaPath = "./data/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        network, class_names, class_colors = darknet.load_network(configPath,  metaPath, weightPath, batch_size=1)
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                            
                except TypeError:
                    pass
            print(namesList)
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./baru.mp4") #perlu diubah sesuai nama file yang diinginkan
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    print("Video Reolution: ",(frame_width, frame_height))

    out = cv2.VideoWriter(
            "./Hasil.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10.0,
            (new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(1/(time.time()-prev_time))
        # cv2.imshow('Demo', image)
        # cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")
    print(objek)

if __name__ == "__main__":
    YOLO()
