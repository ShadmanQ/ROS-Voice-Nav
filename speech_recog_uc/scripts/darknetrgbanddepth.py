#! /usr/bin/env python3

# Authors: Jaime Canizales and Daniel Mallia 
# Date Begun: 8/29/2019
# Based on the code found at the following websites, code written by Jaime and on the human detection system written at Hunter:
# https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
# https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.py
# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
# https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/

# Things to implement/improve:


#REMINDERS:
# HANDLING FOR NO IMAGES RECEIVED YET

# Imports
from nobridge import msg_to_RGB, msg_to_depth_f32, depth_f32_to_msg
import math
import cv2 as cv
import numpy as np
import rospy
import sys
import message_filters
import tf
import tf2_ros
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Vector3
#from tf2_geometry_msgs import PoseStamped

####################################################################################################################################################################################
class darknetrgbanddepth:

  def __init__(self, which_network):
    # Subscribers:
    self.image_sub = message_filters.Subscriber("/head_camera/rgb/image_raw", Image, queue_size = 1)#queue_size = 1, buff_size=2**28
    self.depth_sub = message_filters.Subscriber("/head_camera/depth/image", Image,queue_size = 1)
    self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 1, 3, allow_headerless=True)
    self.ts.registerCallback(self.callback)
    
    # Publisher:
    self.depth_Pub = rospy.Publisher("/DepthPixels", Image, queue_size=10)
    self.position_Pub = rospy.Publisher("/ObjectPosition", Vector3, queue_size=10)

    # Net configuration and initialization
    self.conf_Threshold = 0.5  #Confidence threshold
    self.nms_Threshold = 0.4   #Non-maximum suppression threshold
    self.darknet_Input_Dimensions = (416, 416) # Input dimensions
    
    # Load names of classes
    self.classes_File = "../pre-trained_net_info/coco.names"
    self.classes = None
    with open(self.classes_File, 'rt') as f:
      self.classes = f.read().rstrip('\n').split('\n')
    
    if(which_network == 'YOLO'):
      self.model_Configuration = "../pre-trained_net_info/yolov3.cfg"
      self.model_Weights = "../pre-trained_net_info/yolov3.weights"
    elif(which_network == 'TINY'):
      self.model_Configuration = "../pre-trained_net_info/yolov3-tiny.cfg"
      self.model_Weights = "../pre-trained_net_info/yolov3-tiny.weights"
    
    self.net = cv.dnn.readNetFromDarknet(self.model_Configuration, self.model_Weights)

    # Container and Flags:
    self.rgb_Msg = None # Holds messages of type Image
    self.depth_Msg = None # Holds messages of type Image
    self.net_Ready = True # Flag for net ready to process
    self.image_Updated = False # Flag for a new message is ready for processing

  # Very short casllback - reduces thread to only updating when necessary
  def callback(self, msg, depth_Msg):
    #Only update if net is waiting for a new image
    if(self.net_Ready == True):
      self.rgb_Msg = msg
      self.depth_Msg = depth_Msg
      self.image_Updated = True

  # Primary function - 2D object detection with depth readings.
  def net_Process(self):
    window = cv.namedWindow('test', cv.WINDOW_AUTOSIZE)

    while(not rospy.is_shutdown()): # While ROS is running
      while(not self.image_Updated): # Wait while no new image is available to process; else process
        continue
      
      # Update flags
      self.net_Ready = False
      self.image_Updated = False

      # Convert to images - local variables
      net_Image = msg_to_RGB(self.rgb_Msg)
      net_Depth_Image = msg_to_depth_f32(self.depth_Msg)

      # Create a 4D blob from a frame - image, scaling, out dimensions, mean color,swapRB, crop) - modify mean?
      blob = cv.dnn.blobFromImage(net_Image, 1/255, self.darknet_Input_Dimensions, [0,0,0], swapRB=True, crop=False)

      # Sets the input to the network
      self.net.setInput(blob)

      # Runs the forward pass to get output of the output layers
      outs = self.net.forward(self.get_Outputs_Names())
    
      # Remove the bounding boxes with low confidence
      self.postprocess(net_Image, outs, net_Depth_Image)

      # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the 
      # timings for each of the layers(in layersTimes)
      t, _ = self.net.getPerfProfile()
      label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
      cv.putText(net_Image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

      cv.imshow('test', net_Image)
      if(cv.waitKey(3) == ord('q')):
        print("Shutting down")
        break
      self.net_Ready = True

    cv.destroyWindow('test')

  # Get the names of the output layers
  def get_Outputs_Names(self):
    # Get the names of all the layers in the network
    layers_Names = self.net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_Names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

  # Returns depth at approximately the center pixel in the 2D bounding box
  def get_Center_Distance(self, depthFrame, left, top, width, height):
    x = left + (width//2)
    y = top + (height//2)

    center_Distance = depthFrame[y][x]

    return center_Distance, (y,x)

  def get_Total_Average_Distance(self, depthFrame, left, top, width, height):
    count = 0
    total_Distance = 0
    pairs = []
    for x in range(left, left+width+1):
      for y in range(top, top+height+1):
        if(x < 0 or x > 639 or y < 0 or y > 479):
          continue
        
        pairs.append((y, x))
      
        distance = depthFrame[y][x]
        if(math.isnan(distance)):
          continue

        total_Distance += distance
        count += 1
    
    if(count == 0):
      return 0, []

    total_Avg_Distance = total_Distance / count

    return total_Avg_Distance, pairs



  # Draw the predicted bounding box
  def draw_Pred(self, frame, class_Id, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if self.classes:
        assert(class_Id < len(self.classes))
        label = '%s:%s' % (self.classes[class_Id], label)

    #Display the label at the top of the bounding box
    label_Size, base_Line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_Size[1])
    cv.rectangle(frame, (left, top - round(1.5*label_Size[1])), (left + round(1.5*label_Size[0]), top + base_Line), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

  # Remove the bounding boxes with low confidence using non-maxima suppression
  def postprocess(self, frame, outs, depth):
    frame_Height = frame.shape[0]
    frame_Width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_Ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_Id = np.argmax(scores)
            confidence = scores[class_Id]
            if confidence > self.conf_Threshold:
                center_x = int(detection[0] * frame_Width)
                center_y = int(detection[1] * frame_Height)
                width = int(detection[2] * frame_Width)
                height = int(detection[3] * frame_Height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_Ids.append(class_Id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_Threshold, self.nms_Threshold)
    depth_Pixels = []
    count = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        self.draw_Pred(frame, class_Ids[i], confidences[i], left, top, left + width, top + height)
        #distance, pixel_pair = self.get_Center_Distance(depth, left, top, width, height)
        total_Avg_Distance, pixel_List = self.get_Total_Average_Distance(depth, left, top, width, height)
        depth_Pixels.append(pixel_List)
        dist_String = "Detected: " + str(self.classes[class_Ids[i]]) + " at approximately: " + str(total_Avg_Distance) + " from me."
        print(dist_String)

        # Target point calculation:
        if(count == 0 and total_Avg_Distance > 0 and self.classes[class_Ids[i]] == 'person'): # only do this once (for first object detected and sucessfully processed)
          startX, startY, endX, endY = left, top, (left+width), (top+height)
          c_X = 324.6116067028507 # x coordinate of the optical center
          c_Y = 234.4936855757668 # y coordinate of the optical center
          f_X = 528.1860799146633 # x-focal length
          f_Y = 520.4914794103258 # y-focal length
          p_X = 0.0 # x of point p in camera frame
          p_Y = 0.0 # y of point p 
          p_Z = 0.0 # z of point p

          centerX = (startX + endX)//2
          centerY = (startY + endY)//2
          distance = 0
          divider = 0

          for x in range(centerX-10,centerX+10):
            for y in range(centerY-10, centerY+10):
              if  x < 640 and y< 480:
                if depth[y][x]==depth[y][x]:
                  distance = depth[y][x]
                  divider += 1
                  p_X += distance 
                  p_Y += -(x - c_X)*distance/f_X
                  p_Z += -(y - c_Y)*distance/f_Y


          if divider > 0:
            p_X = p_X/divider
            p_Y = p_Y/divider
            p_Z = p_Z/divider
            position = Vector3()
            position.x = p_X
            position.y = p_Y
            position.z = p_Z
            self.position_Pub.publish(position)
            count +=1

    depth_Copy = msg_to_depth_f32(self.depth_Msg)
    target_Depth_Pixels = np.full((480, 640), np.nan, np.float32)
    for obj in depth_Pixels:
      for (y,x) in obj:
        target_Depth_Pixels[y][x] = depth_Copy[y][x]

    target_Depth_Msg = depth_f32_to_msg(target_Depth_Pixels, self.depth_Msg)

    self.depth_Pub.publish(target_Depth_Msg)
 
# 'Main'
if __name__ == '__main__':
    if(len(sys.argv) != 2):
      print('Usage: ./darknetrgbanddepth.py [YOLO/TINY]') # Usage
      sys.exit()

    if(sys.argv[1] != 'YOLO' and sys.argv[1] != 'TINY'): # Invalid cmd line argument
      print('Invalid net selection')
      sys.exit()
    else:
      rospy.init_node('darknetrgbanddepth', anonymous=True) # Launch
      detector = darknetrgbanddepth(sys.argv[1])
      detector.net_Process()
