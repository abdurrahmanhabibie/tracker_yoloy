from ctypes import *
import math
import os
import cv2
import numpy as np
import time
import darknet

def cvDrawBoxes(detections, img):
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
        for box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)
        objek = centroid_dict
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
    cap = cv2.imread("./nama.jpg") #perlu diubah sesuai nama file yang diinginkan
    frame_width = int(cap.shape[1])
    frame_height = int(cap.shape[0])
    new_height, new_width = frame_height // 2, frame_width // 2
    print("Image Reolution: ",(frame_width, frame_height))

    # out = cv2.VideoWriter(
    #         "./Hasil.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10.0,
    #         (new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    # prev_time = time.time()
    # ret, frame_read = cap.read()
    # # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
    # if not ret:
    #     break

    frame_rgb = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
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
    cv2.imwrite("./Hasil.jpg",image)

    cap.release()
    # out.release()
    print(":::Image Write Completed")
    print(objek)

if __name__ == "__main__":
    YOLO()
