'''
Script to test traffic light localization and detection
'''

import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob



cwd = os.path.dirname(os.path.realpath(__file__))


os.chdir(cwd+'/models')
from object_detection.utils import visualization_utils as vis_util


class TLClassifier(object):
    def __init__(self):

        self.signal_classes = ['Red', 'Green', 'Yellow']
       
        self.signal_status = None
        
        self.tl_box = None
        
        os.chdir(cwd)
        
        #keras classification model
        self.cls_model = load_model('tl_model_1.h5')
        
        #tensorflow localization/detection model
       
        detect_model_name = 'ssd_mobilenet_v1_coco_11_06_2017'
        #detect_model_name = 'ssd_inception_v2_coco_11_06_2017'
        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'
        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        
        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize 
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')
               
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')
    # Helper function to convert image into numpy array    
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)       
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
    
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)       
        
    def get_localization(self, image, visual=False):  
        """Determines the locations of the traffic light in the image

        Args:
            image: camera image

        Returns:
            list of integers: coordinates [x_left, y_up, x_right, y_down]

        """
        category_index={1: {'id': 1, 'name': u'person'},
                        2: {'id': 2, 'name': u'bicycle'},
                        3: {'id': 3, 'name': u'car'},
                        4: {'id': 4, 'name': u'motorcycle'},
                        5: {'id': 5, 'name': u'airplane'},
                        6: {'id': 6, 'name': u'bus'},
                        7: {'id': 7, 'name': u'train'},
                        8: {'id': 8, 'name': u'truck'},
                        9: {'id': 9, 'name': u'boat'},
                        10: {'id': 10, 'name': u'traffic light'},
                        11: {'id': 11, 'name': u'fire hydrant'},
                        13: {'id': 13, 'name': u'stop sign'},
                        14: {'id': 14, 'name': u'parking meter'}}  
        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})
          
              if visual == True:
                  vis_util.visualize_boxes_and_labels_on_image_array(
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,min_score_thresh=.2,
                      line_thickness=3)
    
                  plt.figure(figsize=(9,6))
                  plt.imshow(image)
                  plt.show()  
              
              boxes=np.squeeze(boxes)
              classes =np.squeeze(classes)
              scores = np.squeeze(scores)
    
              cls = classes.tolist()
              #print(cls)
              # Find the first occurence of traffic light detection id=10
              idx = next((i for i, v in enumerate(cls) if v == 10.), None)
              # If there is no detection
              if idx == None:
                  box=[0, 0, 0, 0]
                  print('no detection!')
              # If the confidence of detection is too slow, 0.3 for simulator    
              elif scores[idx]<=0.02:
                  box=[0, 0, 0, 0]
                  print('low confidence:', scores[idx])
              #If there is a detection and its confidence is high enough    
              else:
                  #*************corner cases***********************************
                  dim = image.shape[0:2]
                  box = self.box_normal_to_pixel(boxes[idx], dim)
                  box_h = box[2] - box[0]
                  box_w = box[3] - box[1]
                  ratio = box_h/(box_w + 0.01)
                  # if the box is too small, 20 pixels for simulator
                  if (box_h <10) or (box_w<10):
                      box =[0, 0, 0, 0]
                      print('box too small!', box_h, box_w)
                  # if the h-w ratio is not right, 1.5 for simulator    
                  elif (ratio<1.5):
                      box =[0, 0, 0, 0]
                      print('wrong h-w ratio', ratio)
                  else:    
                       print(box)
                       print('localization confidence: ', scores[idx])
                 #****************end of corner cases***********************      
              self.tl_box = box
             
        return box
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): cropped image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # Resize cropped 
        #img_resize = cv2.resize(image, (32, 32)) 
        img_resize = image
        # Color map conversion
        img_resize=cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB) 
        # Convert to four-dimension input as required by Keras
        img_resize = np.expand_dims(img_resize, axis=0).astype('float32')
        # Normalization
        img_resize/=255.
        # Prediction
        predict = self.cls_model.predict(img_resize)
        predict = np.squeeze(predict, axis =0)
        # Get color classification
        tl_color = self.signal_classes[np.argmax(predict)]
        print(tl_color,', Classification confidence:', predict[np.argmax(predict)])
        
        # TrafficLight message
        self.signal_status = tl_color
        
        
        return self.signal_status

if __name__ == '__main__':
        
        tl_cls =TLClassifier()
        os.chdir(cwd)
        TEST_IMAGE_PATHS= glob(os.path.join('traffic_light_images/', '*.jpg'))
        
        for i, image_path in enumerate(TEST_IMAGE_PATHS):
            print('')
            print('*************************************************')
            
            img_full = Image.open(image_path)
            img_full_np = tl_cls.load_image_into_numpy_array(img_full)
            img_full_np_copy = np.copy(img_full_np)
            start = time.time()
            b = tl_cls.get_localization(img_full_np, visual=False)
            end = time.time()
            print('Localization time: ', end-start)
            start = time.time()
            # If there is no detection or low-confidence detection
            if np.array_equal(b, np.zeros(4)):
               print ('unknown')
               plt.figure(figsize=(9,6))
               plt.imshow(img_full_np)
               plt.show()
            else:    
               cv2.rectangle(img_full_np,(b[1],b[0]),(b[3],b[2]),(0,255,0),2)
               plt.figure(figsize=(9,6))
               plt.imshow(img_full_np)
               plt.show()
               img_np = cv2.resize(img_full_np_copy[b[0]:b[2], b[1]:b[3]], (32, 32)) 
               plt.figure(figsize=(5,5))
               plt.imshow(img_np)
               plt.show()
               tl_cls.get_classification(img_np)
              
            end = time.time()   
            
