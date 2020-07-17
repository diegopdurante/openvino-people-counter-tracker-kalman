#!/usr/bin/env python3
"""
Helper image functions
"""

import numpy as np
import cv2
import sys
import csv

__author__ = "Diego Patricio Durante"
__copyright__ = ""
__credits__ = ["Diego Patricio Durante"]

__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Diego Patricio Durante"
__email__ = ""
__status__ = ""

def send_to_stdout(image):
    """
    Send output to stdout for ffmpeg
    :param: image, image to send to stdout
    :return: None
    """
    
    # Stack the mask so FFmpeg understands it
    # Our operations on the frame come here    
    b,g,r = cv2.split(image)

    out_mask = np.dstack((b, g, r))
    out_mask = np.uint8(out_mask)

    sys.stdout.buffer.write(out_mask)
    sys.stdout.flush()
    
def make_input_blob(frame, final_shape):
    """
    Make input blob according to inference input shape
    # TODO: Only works for square shapes. Will fail if aspect ratio for neural network is different to 1:1
    : param: #TODO
    :return: blob, scales
    :trype: #TODO
    """
    # TODO: Take in account different aspect ratios and image dimmensions (as example, gray)
    
    w = frame.shape[1]
    h = frame.shape[0]
    
    w_final = final_shape[3]
    h_final = final_shape[2]
    
    crop = max(frame.shape[0], frame.shape[1])
    frame_new = np.zeros((crop, crop, 3), dtype = frame.dtype)
    offset_y = int((crop - h)/2.0)
    offset_x = int((crop - w)/2.0)

    frame_new[offset_y:offset_y + h, offset_x:offset_x + w, :] = frame
    
    p_frame = cv2.resize(frame_new, (w_final, h_final))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    
    scale_x = (1.0*crop/frame.shape[1])
    scale_y = (1.0*crop/frame.shape[0])
    
    return p_frame, scale_x, scale_y, offset_x, offset_y
    
    # TODO: If input aspect ratio is not square, I will correct differently. Work in progress (no practical case for now)
    # En el caso de que el aspect ratio final no fuera cuadrado, sería algo así
#     aratio = float(w)/frame.shape[0]
#     aratio_final = float(w_final)/h_final
    
#     size_x = 0
#     size_y = 0
#     offset_x = 0
#     offset_y = 0
    
#     if aratio_final > aratio:
#         # El y estará completo, el x no
#         size_y = h
#         size_x = int(h*aratio_final)
#         offset_x = int((size_x - frame.shape[1])/2.0)
#     else:
#         # El x estará completo, el y no
#         size_x = w
#         size_y = int(w/aratio_final)
#         offset_y = int((size_y - frame.shape[0])/2.0)
        
#     frame_new = np.zeros((size_y, size_x, 3), dtype = frame.dtype)
#     frame_new[offset_y:offset_y + frame.shape[0], offset_x:offset_x + frame.shape[1], :] = frame

#     p_frame = cv2.resize(frame_new, (w_final, h_final))
#     p_frame = p_frame.transpose((2,0,1))
#     p_frame = p_frame.reshape(1, *p_frame.shape)
    
    # # TODO: Uncomplete
#     scale_x = (1.0)#*crop/frame.shape[1])
#     scale_y = (1.0)#*crop/frame.shape[0])
    
#     return p_frame, scale_x, scale_y, offset_x, offset_y


def correct_scale(box_p1, box_p2, scale_x, scale_y, offset_x, offset_y):
    """
    Correct prediction scales
    :return: Corrected points for bounding boxes
    """
    return tuple(np.array([box_p1[0]*scale_x - offset_x, box_p1[1]*scale_y - offset_y]).astype(int)),\
            tuple(np.array([box_p2[0]*scale_x - offset_x, box_p2[1]*scale_y - offset_y]).astype(int)) 

def input_is_image(inp):
    """
    Check if input is image or not
    :param: String containing input name
    :return: True if extension of input is an image, else False
    """
    
    extensions_image = ["jpg", "png"]
    
    if inp.split(".")[-1] in extensions_image:
        return True
    else:
        return False

def get_video_capture(inp):
    """
    Set VideoCapture source
    :return: VideoCapture object initialized
    """
    
    cap = None
    
    if inp == "CAM":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(inp)

    return cap
    


def add_info_to_image(image, count, total_count, frame_number):
    """
    This function appends text information to image
    :param: #TODO
    :return: None
    """
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.75

    # White
    color = (255, 255, 255) 

    # Line thickness in px
    thickness = 2

    image = cv2.putText(image, "Count: " + str(count), (10, 30), font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
    
    image = cv2.putText(image, "Total count: " + str(total_count), (10, 60), font,  
                       fontScale, color, thickness, cv2.LINE_AA)
    
    image = cv2.putText(image, "Frame: " + str(frame_number), (10, 90), font,  
                       fontScale, color, thickness, cv2.LINE_AA)


def plot_detection(image, box_p0, box_p1, color_rect, radius_circle, color_circle, rect_width = 1, text = ""):
    """
    Plot one detection on image and optionally a text
    :param: #TODO
    :return: None
    """
    
    center_x = int((box_p0[0] + box_p1[0])/2.0)
    center_y = int((box_p0[1] + box_p1[1])/2.0)
    
    filled = -1 + 2*(radius_circle > 0)
    radius_circle = abs(radius_circle)

    cv2.circle(image, (center_x, center_y), radius_circle, color_circle, filled)

    cv2.rectangle(image, box_p0, box_p1, color_rect, rect_width)
    
    if len(text)!= 0:
        cv2.putText(image, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        
def process_output(output, image, num_frame, scale_x, scale_y, offset_x, offset_y, tracker = None, prob_threshold = 0.6, class_to_detect = 1, debug = False):
    """
    Process output from inference
    :param: #TODO
    :return: # TODO: people counts on image, duration of each new detection, and total count of detections from the start
    """
    width = image.shape[1]
    height = image.shape[0]
    
    # SSD networks tested (models converted with 100 bounding boxes and 200 bounding boxes)
    if output.shape == ((1,1,200,7)) or output.shape == ((1,1,100,7)):    
        # I supose that it will have each box with this information:
        # [image_id, label, conf, x_min, y_min, x_max, y_max]
        
        if debug:
            print("shape resized ", image.shape)
            print("scale x", scale_x)
            print("scale y", scale_y)
            print("scale x", width)
            print("scale x", height)
        
        # Get true predictions according to setup
        pred_thres = output[0,0,:,2] > prob_threshold
        pred_class = output[0,0,:,1] == class_to_detect
        pred_and = pred_thres & pred_class
        
        if debug:
            print( output[:,:,pred_and][0][0].shape)
        
        # Initialize empty data to be filled an returned
        count = 0
        total_count = 0
        durations = []
        
        # List to update tracked objects
        new_boxes = []
        
        # If they are some detections, update them
        if pred_and.sum() != 0:
            for i in output[:,:,pred_and][0][0]:
                box_p0 = tuple((i[3:5]*(width, height)).astype(int))
                box_p1 = tuple((i[5:7]*(width, height)).astype(int))
                
                # Correct box scale
                box_p0, box_p1 = correct_scale(box_p0, box_p1, scale_x, scale_y, offset_x, offset_y)

                plot_detection(image, box_p0, box_p1, (0,255,0), -10, (0, 255, 0), 5)
                
                # Save new detections
                new_boxes.append([box_p0[0], box_p0[1], box_p1[0], box_p1[1]])
                
                # Plot information about probability
                box_p3 = (int(box_p0[0] + (box_p1[0] - box_p0[0])*i[2]), box_p0[1] + 20)
                cv2.rectangle(image, box_p0, box_p3, (255 * i[2], 0, 0), -5)

                if debug:
                    print("Box: ", box_p0, box_p1)
                    print("Confidence: ", i[2])
                    print("Class: ", i[1])
                
                # TODO: Improve
                # Debug: To recreate detection after and trace tracker behavior
                write_csv = True
                if write_csv:
                    with open('boxes.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow([num_frame] + list(box_p0) + list(box_p1))
                    
        # Update tracker parameters to improve detection
        if tracker is not None:
            # Update predicted possitions and associate detections 
            tracker.update(new_boxes)
            
            # Plot updated predictions
            for idx, box in tracker.get_boxes_and_ids():

                pt1 = (box[0], box[1])
                pt2 = (box[2], box[3])

                plot_detection(image, pt1, pt2, (255,0,255), -5, (255, 0, 255), 2, str(idx))

            # Get data about people counted from tracker
            count = tracker.count_objects()
            total_count = tracker.count_total()
            durations = tracker.get_non_informed_times()
            
            if debug == True:
                print("Actual: ", count, total_count)
                tracker.print_objects()
                
        # Append statistical information to image
        add_info_to_image(image, count, total_count, num_frame)
        
        return count, durations, total_count

    # Error, network not handled 
    else:
        if debug:
            print(output.shape)
            print("Incorrect output")
#             print(output)
        quit()
