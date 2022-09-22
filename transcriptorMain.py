#################################################################################################
#
#   Project:        Transcriptor
#   File:           main
#   Author:         Marcos Fernandez
#   Date:           September 2017
#
#   Description:    Main routine
#
#################################################################################################

#import packages
import cv2
import argparse
import datetime
import imutils
import time
import numpy as np
import json
import requests
from mimetypes import guess_type
from pathlib import Path
import os

def transcriptorMain(image_old, image_new, models = None):

    corner_points=None
    backgroundTuned=None
    noiseThreshold=None
    removePeople=None

    # Crop image
    if (models == 'warning_sign'):
        print("Warning Sign inference")
        img_splits = crop(image_new)
    else:
        print("Anticlimbing inference")
        image_resized = resize(image_new, split_height=1024, split_width=1024)
        img_splits = [[image_resized]]


    # Detect objects & Invocar API
    object_detected = False
    img_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
    
    corner_points_new = None
    img_result_column = image_new.copy()
    img_result_row = image_new.copy()

    print("Inference started...")
    start = time.time()

    i_nx = 0
    j_nx = 0
    first_column = True
    first_row = True
    for i in img_splits:
        j_nx = 0
        #print("Split: ", i_nx)
        for j in i:
            #print("Split: ", j_nx)
            #print(filename_str)
            filename_str = str(os.path.join(img_save_dir, "0001_"+str(j_nx) + "_" + str(i_nx)+".jpg"))
            cv2.imwrite(filename_str, j) #, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            corner_points_new = None
            
            #try:
            infer_result = infer(filename_str)
            #print('Return prediction: ', infer_result)
            #print('Predictions qty: ', len(infer_result['data']['predictions']))
            for predictionIt in infer_result['data']['predictions']:

                prediction = infer_result['data']['predictions'][predictionIt]

                corner_points_new = [int(prediction['coordinates']['xmin']),int(prediction['coordinates']['ymin']),int(prediction['coordinates']['xmax']),int(prediction['coordinates']['ymax'])]
                
                corner_points_round = np.round(corner_points_new,decimals=3)
                corner_points_round = corner_points_round.astype(np.int64)
                start_point = (corner_points_round[0], corner_points_round[1])
                end_point = (corner_points_round[2], corner_points_round[3])

                #cv2.drawContours(image_copy, [corner_points_round], -1, (0, 255, 0), 2)
                j = cv2.rectangle(j, start_point, end_point, (0, 255, 0), 2)
                #print("Prediction: ", str(prediction), " in ", str(j_nx), ", ", str(i_nx) )
                object_detected = True

            if first_column is True:
                img_result_column = j.copy()
                first_column = False
            else:
                img_result_column = cv2.hconcat([img_result_column, j])
            #print("j: ", j_nx)
            j_nx += 1
        
        #print("i: ", i_nx)
        i_nx += 1
        first_column = True
        if first_row is True:
            img_result_row = img_result_column.copy()
            first_row = False
        else:
            img_result_row = cv2.vconcat([img_result_row, img_result_column])

    img_result = img_result_row.copy()

    end = time.time()
    elapsed = end - start
    print("Inference completed in ", str(elapsed), " seconds.")

    if object_detected is False:
        height = img_result.shape[0]
        width = img_result.shape[1]

        x, y, w, h = 10, 10, width - 20, height - 20
        sub_img = img_result[y:y+h, x:x+w]
        red_rect = np.zeros(sub_img.shape, np.uint8) #np.ones(sub_img.shape, dtype=np.uint8) * 255
        red_rect = cv2.rectangle(red_rect, (10,10), (red_rect.shape[1]-10, red_rect.shape[0]-10), (0, 0, 255), cv2.FILLED)
        res = cv2.addWeighted(sub_img, 1.0, red_rect, 0.25, 1.0)

        #print("Size result: ", width, " x ", height)
        #print("Size result sub: ", sub_img.shape[1], " x ", sub_img.shape[0])
        
        # Putting the image back to its position
        img_result[y:y+h, x:x+w] = res
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img_result, "NOT FOUND", (20, height-20), font, (1.0)*(height/512), (0, 0, 255), 7)

    else:
        height = img_result.shape[0]
        width = img_result.shape[1]

        x, y, w, h = 10, 10, width - 20, height - 20
        sub_img = img_result[y:y+h, x:x+w]
        red_rect = np.zeros(sub_img.shape, np.uint8) #np.ones(sub_img.shape, dtype=np.uint8) * 255
        red_rect = cv2.rectangle(red_rect, (0, 0), (red_rect.shape[1]-10, red_rect.shape[0]-10), (0, 255, 0), cv2.FILLED)
        res = cv2.addWeighted(sub_img, 1.0, red_rect, 0.25, 1.0)

        # Putting the image back to its position
        img_result[y:y+h, x:x+w] = res
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img_result, "FOUND", (20, height-20), font, (1.0)*(height/512), (0, 0, 255), 7)

        #print("Size result: ", width, " x ", height)
        #print("Size result sub: ", img_result.shape[1], " x ", img_result.shape[0])
        #print("Size result sub: ", sub_img.shape[1], " x ", sub_img.shape[0])
        #print("Size result sub: ", red_rect.shape[1], " x ", red_rect.shape[0])
        #print("Size result sub: ", res.shape[1], " x ", res.shape[0])

        #except:
        #print("Error: ")
    
    return img_result, corner_points_new 


def infer(filename):
    ##url = "http://ec2-3-120-228-112.eu-central-1.compute.amazonaws.com:8080/api/v1/images"
    url = "http://127.0.0.1:8080/api/v1/images"
    #with open(Path(str(filename)).resolve(), "rb") as f:
    with open(filename, "rb") as f:
        files = [("file", (Path(filename).name, f, guess_type(filename)))]
        response = requests.request("POST", url, files=files)
    #print(response.text)
    return json.loads(response.text)


def crop(img, split_width=412, split_height=412):
    # Compute the amount of images
    img_h, img_w, _ = img.shape
    Y_points = start_points(img_h, split_height, 0.0)
    X_points = start_points(img_w, split_width, 0.0)
    #print("X: ",X_points)
    #print("Y: ", Y_points)

    #Split and storage
    #img_splits = np.empty([len(Y_points), len(X_points) + 1], dtype=int)
    #img_splits = np.empty([len(Y_points), 0], dtype=int)   
    rows = len(Y_points)
    columns= len(X_points)
    img_splits = [[0 for x in range(columns)] for x in range(rows)]
    #print("Length splits: ", len(img_splits), " X ", len(img_splits[0]))
    #print("im_splits: ", img_splits)
    #print(rows)
    #print(columns)
    #print(len(img_splits))
    #print(len(img_splits[0]))
    row_indx = 0
    col_indx = 0
    for i in Y_points:
        col_indx = 0
        for j in X_points:
            column_width = split_width
            column_height = split_height
            if X_points[col_indx+1] == 99999: #last column in image, so give full size
                column_width = img_w - j
            if Y_points[row_indx+1] == 99999:
                column_height = img_h - i
            split = img[i:i + column_height, j:j + column_width]
            img_splits[row_indx][col_indx] = split.copy()
            #print("C: ", col_indx)
            #print("R: ", row_indx)
            if X_points[col_indx+1] == 99999:
                img_splits[row_indx].pop(col_indx+1)
                break
            col_indx += 1
        if Y_points[row_indx+1] == 99999 and X_points[col_indx+1] == 99999:
            img_splits.pop(row_indx+1)
            break
        row_indx += 1

    return img_splits

def start_points(size, split_size, overlap=0.0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            #points.append(size - split_size)
            points.append(99999) #end of file
            break
        else:
            points.append(pt)
        counter += 1
    return points

def resize(img, split_height=256, split_width=256):
    img_h, img_w, _ = img.shape

    new_height = img_h
    new_width = img_w

    if (img_h > split_height) or (img_w > split_width):
        if (img_h > img_w):
            new_height = split_height
            new_width = int((img_w/img_h)*new_height)
        else:
            new_width = split_width
            new_height = int((img_h/img_w)*new_width)
    
    dsize = (new_width, new_height)

    img = cv2.resize(img, dsize)

    return img


