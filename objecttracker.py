#!/usr/bin/env python3
"""
Multiple Object Tracker from predictions using
* Kalman Filters to predict next positions on each iteration for each object,
* Hungariam algorithm to re associate detections
correct predictions.
* IoU to correlate predictions and detections also.
"""

import numpy as np
from munkres import Munkres, print_matrix

import time
import cv2

__author__ = "Diego Patricio Durante"
__copyright__ = ""
__credits__ = ["Diego Patricio Durante"]

__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Diego Patricio Durante"
__email__ = ""
__status__ = ""

class ObjectTracker:
    """
    Multiple object tracker.
    This object tracker uses:
    * Kalman Filters to predict next positions on each iteration for each object,
    * Hungariam algorithm to re associate detections
    correct predictions.
    * IoU to correlate predictions and detections also.
    """
    
    def __init__(self, all_max_life = 5, iou_th = 0.2, noise_covariance = 0.3):
        """
        Initialize tracker
        :param: all_max_life: quantity of epochs
        :param: iou_th: threshold for intersection over union to make associations
        :param: noise_covariance: Noise covariance for Kalman Filter
        """
        
        self.max_life = all_max_life

        self.next_id = 1

        self.detections = None

        self.objects = dict()
        self.objects_life = dict()

        # To calculate time taken by object
        self.objects_time = dict()       
        self.non_informed_times = []

        self.iou_thres = iou_th

        # One kalman filter by object
        self.filters = dict()
        
        self.noise_cov = noise_covariance
        
        # TODO: Idea: Agregar variable de min_to_be_real (cantidad de aciertos consecutivos para ser real?)

    def new_object(self, x1, y1, x2, y2):
        """
        Add a new object to track
        """
        new_detection = [x1, y1, x2, y2]
        self.objects[str(self.next_id)] = new_detection
        self.objects_life[str(self.next_id)] = self.max_life
        
        # Starting time for each object
        self.objects_time[str(self.next_id)] = time.time()

        # Initializw filter for each object
        self.filters[str(self.next_id)] = cv2.KalmanFilter(4,2)
        self.filters[str(self.next_id)].measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32) # mediciones (x, y)
        self.filters[str(self.next_id)].transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32) # Como cambian los estados (x + dx, y + dy)
        self.filters[str(self.next_id)].processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * self.noise_cov
        
        # Reset initial status of each filter (this will be improved)
        self.reset_status(str(self.next_id), x1, y1, x2, y2)

        self.next_id = self.next_id + 1
        
    def remove_object(self, key):
        """
        Delete some tracked object
        :param: key: Key of tracked object
        """
        del self.objects_life[key]
        del self.objects[key]
        self.non_informed_times.append(time.time() - self.objects_time[key])
        del self.objects_time[key]
        del self.filters[key]

    def reset_status(self, key, x1, y1, x2, y2):
        """
        Restart some kalman filter for initial status
        # TODO:Pending to complete...
        """
        xc = int((x1 + x2)/2)
        yc = int((y1 + y2)/2)

        center = np.array([xc,yc], np.float32)
        for i  in range(100):
            self.filters[key].correct(center)
            self.filters[key].predict()

    def get_iou(self, box1, box2):
        """
        Solve Intersection over union algorithm
        :param: box1: first box to intersect
        :param: box2: second box to intersect
        :return: IoU value, it will be between 0 and 1 
        """
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate area of intersection
        i = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))
        if i == 0:
            return 0

        u1 = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
        u2 = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

        iou = i / float(u1 + u2 - i)

        return iou

    def get_distances(self, boxes):
        """
        Get distance matrix, to solve association problem
        :param: Boxes for detections
        :return: Distance matrix
        """
        results = np.zeros((len(boxes), len(self.objects)))
        for idx1, b in enumerate(boxes):
            # line = [] # Initialize correspondences for this object
            for idx2, o in enumerate(self.objects):
                # line.append(get_iou(n, o))
                results[idx1, idx2] = self.get_iou(b,
                                                   self.objects[str(o)])
                # print(self.objects[str(o)])
            # results.append(line)

        return results

    def update(self, boxes):
        """
        Update predictions, associate with detections and add new objects if have no association
        Also delete dead objects
        :param: Detected boxes
        :return: None
        """
        non_asociated = [i for i in range(len(boxes))]

        pred = None

        # Update prediction for each tracked object
        for key in self.filters:
            pred = self.filters[key].predict()

            # Update bounding box according to xc, yc predicted and size measured
            new_xc = pred[0]
            new_yc = pred[1]

            xc = float(self.objects[key][0]+self.objects[key][2])/2
            yc = float(self.objects[key][1]+self.objects[key][3])/2

            half_w = self.objects[key][2] - xc
            half_h = self.objects[key][3] - yc

            # Update object boxes
            self.objects[key] = [int(new_xc - half_w), int(new_yc - half_h), int(new_xc + half_w), int(new_yc + half_h)]

        if len(boxes) != 0 and len(self.objects) != 0:
            m = Munkres()

            # Get distances between detections and predictions to match
            distances = self.get_distances(boxes)

            # assert len(self.objects) == len(self.objects_time) == len(self.objects_life) == len(self.filters), "Bug on code"

            # It seems to have a bug when matrix is a numpy array of shape (2,1), by that I convert to list
            indices = m.compute((-distances).tolist())
#             print("Válidos: ", indices)

            # For each match possible match, filter impossible matches
            for i, j in indices:
            
                # Check for valid associations
                if distances[i, j] > self.iou_thres:
#                     print("cumplen", i, j)
                    key = list(self.objects.keys())[j]
                    self.objects_life[key] = self.max_life

                    xc = float(boxes[i][0]+boxes[i][2])/2
                    yc = float(boxes[i][1]+boxes[i][3])/2

                    meas_centers = np.array([xc,yc], np.float32)
                    
                    # If there is a match, feedback to kalman filter
                    self.filters[key].correct(meas_centers)

                    # Actualizo el ancho, esto debería ser predicho pero no está contemplado en el filtro de kalman
                    # Hago de cuenta que la actualización está bien
                    # Calculo el ancho del bonding box medido
                    half_w = boxes[i][2] - xc
                    half_h = boxes[i][3] - yc

                    # Mantengo el centro predicho
                    xc = float(self.objects[key][0]+self.objects[key][2])/2
                    yc = float(self.objects[key][1]+self.objects[key][3])/2

                    self.objects[key] = [int(xc - half_w), int(yc - half_h), int(xc + half_w), int(yc + half_h)]                 

                    # Mark association as valid deleting to non associated detections
                    non_asociated.remove(i)

        # Add non associated values as new objects
        for i in non_asociated:
            # print(boxes[i])
            self.new_object(boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3])

        keys_to_del = []
        for key in self.objects_life:
            # If a tracked object was dissapeared for last iterations (no life), delete it
            if self.objects_life[key] <= 0:
                keys_to_del.append(key)

            # Else decrement life
            else:
                self.objects_life[key] = self.objects_life[key] - 1

        # Delete pending asociations
        for key in keys_to_del:
            self.remove_object(key)

    def print_objects(self):
        """
        Function to know actual status, for debug purposes
        """
        
        print("Next id: ", self.next_id)
        for key in self.objects:
            print(key, ": ", self.objects[key], self.objects_life[key])
            
    def get_centers(self):
        """
        Get objects information (Centers of tracked objects)
        :return: Number of objects alive on last update
        """
        
        centers = []
        
        for key in self.objects:
            value = self.objects[key]
            xc = int((value[0] + value[2])/2)
            yc = int((value[1] + value[3])/2)
            centers.append((xc, yc))
            
        return centers
            
    def get_boxes_and_ids(self):
        """
        Get objects information (Boxes and ids of tracked objects)
        :return: Number of objects alive on last update
        """
        
        retval = []
        
        for key in self.objects:
            value = self.objects[key]
            xc = int((value[0] + value[2])/2)
            yc = int((value[1] + value[3])/2)
            retval.append([key, self.objects[key]])

        return retval
    
    def get_centers_and_ids(self):
        """
        Get objects information (Centers and ids of tracked objects)
        :return: Number of objects alive on last update
        """
        
        retval = []
        
        for key in self.objects:
            value = self.objects[key]
            xc = int((value[0] + value[2])/2)
            yc = int((value[1] + value[3])/2)
            retval.append([key, (xc, yc)])

        return retval

    def get_centers_boxes_and_ids(self):
        """
        Get objects information (Centers, boxes and ids of tracked objects)
        :return: Number of objects alive on last update
        """
        
        retval = []
        
        for key in self.objects:
            value = self.objects[key]
            xc = int((value[0] + value[2])/2)
            yc = int((value[1] + value[3])/2)
            retval.append([key, (xc, yc), self.objects[key]])

        return retval
    
    def count_objects(self):
        """
        Get number of objects alive
        :return: Number of objects alive on last update
        """
        return len(self.objects)
    
    def count_total(self):
        """
        Get number of objects from start
        :return: Number of objects from start
        """
        return self.next_id - 1
    
    def get_non_informed_times(self):
        """
        Get number of objects dead and empty buffer of object epochs
        :return: list with epochs for dead detections
        """
        to_return = self.non_informed_times.copy()
        self.non_informed_times = []
        return to_return

# Ejemplo
if __name__ == "__main__":
    # Paso 1, instancio el tracker
    Tracker = ObjectTracker()

    # Iterativamente agrego valores y actualizo
    l = [[2,2, 9, 9], [3,4, 10, 1], [9,9, 11, 11]]
    Tracker.update(l)

    Tracker.print_objects()

    m = [[3,3,10,10], [3,3,10,11]]
    Tracker.update(m)

    Tracker.print_objects()

    n = [[4,3,10,10], [3,3,10,11]]

    Tracker.update(n)

    Tracker.print_objects()
    Tracker.update(n)

    Tracker.print_objects()
    Tracker.update(n)

    Tracker.print_objects()
    Tracker.update(n)

    Tracker.print_objects()
