"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
import time
import socket
import json
import cv2
import os

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import numpy as np
from objecttracker import ObjectTracker

from image_helpers import send_to_stdout, make_input_blob, input_is_image, add_info_to_image, process_output, get_video_capture

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
# TODO: If input from camera

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file (if it is CAM, try to get frames from camera on idx 0)")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    
    parser.add_argument("-dbg", "--debug", type=bool, default=False,
                        help="Print debug outputs and doesn't stream frames to stdout"
                        "(False by default)")    
    parser.add_argument("-w", "--wait_ms", type=int, default=30,
                        help="Delay (ms) to take another frame"
                        "(30 by default)")
    
    parser.add_argument("-s", "--skip_first_frames", type=int, default=-1,
                        help="Strip first frames"
                        "(None by default)")
    
    parser.add_argument("-lp", "--loop", type = bool, default = False,
                        help="Process in loop"
                        "(Only for video, no by default)")
    
    parser.add_argument("-c", "--class_to_detect", type = int, default = 1,
                        help="Class to detect"
                        "(1 by default)")
    
    return parser


def connect_mqtt():
    # Connect to the MQTT client
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

    
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # For debug purposes, delete boxes.csv file, to test tracker after wiothout make inferences again
    try:
        os.remove('boxes.csv')
    except:
        pass
    
    debug = args.debug
    
    # Set an instance of inference object
    infer_network = Network()
    
    # Set an instance of object tracker
    tracker = ObjectTracker(10)
    
    # Set Probability threshold for detections and class
    prob_threshold = args.prob_threshold
    class_to_detect = args.class_to_detect
    
    # Frame will be the image to infer
    frame = None

    # Load the model through `infer_network`
    load_model_result = infer_network.load_model(args.model, args.device, args.cpu_extension)
    if load_model_result == False:
        print("Unable to initialize network")
        return
    
    # Handle the input stream
    image_name = args.input
    single_image_mode = input_is_image(image_name)
    
    if single_image_mode == True:
        # Input is a single image
        frame = cv2.imread(args.input)
        
        width = frame.shape[1]
        height = frame.shape[0]
        
    else:
        # Input is a video stream
        cap = get_video_capture(image_name)
        cap.open(image_name)
    
        width = int(cap.get(3))
        height = int(cap.get(4))
        
    if debug:
        print(width, height)
        
        
    # Get final dimensions
    final_shape = infer_network.get_input_shape()
    
    if args.debug == True:
        print(final_shape)
    
    # Constantly defined (for ffmpeg)
    width_out = 768
    height_out = 432

    # Send message troutgh mqtt to initialize ui
    client.publish("person", json.dumps({"count": 0}))

    num_frame = 0
    
    # Skip first frames if needed
    strip_until = args.skip_first_frames
    while num_frame < strip_until:
        if single_image_mode == False:
            flag, frame = cap.read()
            if not flag:
                return
            num_frame = num_frame + 1

    # Loop until stream is over or infinitely if loop is enabled
    while True:

        # Read from the video capture
        if single_image_mode == False:
            flag, frame = cap.read()
            if not flag:
                if args.loop == True:
                    cap = get_video_capture(image_name)
                    cap.open(image_name)
                    continue
                else:
                    break
                    
            if debug == True:
#                 key_pressed = cv2.waitKey(1)
                pass
            else:
                key_pressed = cv2.waitKey(args.wait_ms) # No es necesario
            
#         frame = frame[300:660, 300:940, :]

        # Pre-process frame the according to output parameters
        frame_plot = cv2.resize(frame, (width_out, height_out))

        # Pre-process frame for inference
        p_frame, scale_x, scale_y, offset_x, offset_y = make_input_blob(frame, final_shape)
        
        # Start asynchronous inference for specified request 
        status = infer_network.exec_net(p_frame)
        
        # Wait until inference is made (blocking operation)
        if infer_network.wait() == 0:
            
            # Get the results of the inference request 
            out_inference = infer_network.get_output()
            
            # For models with multiple output layers:
#             out_inference = infer_network.get_outputs()
#             for out in out_inference:
#                pass # Do something with each layer

            # Extract any desired stats from the results
            count, non_informed_avg, total_count = process_output(out_inference, frame_plot, num_frame, scale_x, scale_y, offset_x, offset_y, tracker, prob_threshold, class_to_detect, debug)

            if debug:
                print("Persons detected: ", count)
                print("Time average: ", non_informed_avg)
                
            # Publish relevant information for ui
            # current_count, total_count and duration to the MQTT server
            # Topic "person": keys of "count" and "total"
            # Topic "person/duration": key of "duration"
            
            for each_one in non_informed_avg:
                client.publish("person/duration", json.dumps({"duration": each_one}))
    
            if len(non_informed_avg) > 0:
                client.publish("person", json.dumps({"total": total_count})) # ui is not processing it (it should)

            # Every some frames, inform peoples in actual frame (to avoid unnecesary overload on MQTT channels)
            if num_frame%20 == 0:
                client.publish("person", json.dumps({"count": count}))
                
        # Send the frame to the FFMPEG server
        if not debug:
            send_to_stdout(frame_plot)

        ### Write an output image if `single_image_mode`
        if single_image_mode == True:
            filename = 'output_image.jpg'
            cv2.imwrite(filename, frame_plot)
            if debug:
                print("{} wroten".format(filename))
            break
        
        # Update frame index
        num_frame = num_frame + 1


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    
    # Grab command line args
    args = build_argparser().parse_args()
    
    # Connect to the MQTT server
    client = connect_mqtt()
    
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    """
    Call main function.
    :return: None
    """
    main()
