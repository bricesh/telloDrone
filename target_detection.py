import numpy as np
import os
import tensorflow as tf
import cv2

cap = cv2.VideoCapture(1)

# Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

# Model preparation 
MODEL_NAME = 'ssd_mobilenet_v1_coco'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'  # Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      norm_target = (boxes[0][0][1]+(boxes[0][0][3]-boxes[0][0][1])/2, boxes[0][0][0]+(boxes[0][0][2]-boxes[0][0][0])/2)
      # Visualization of the results of a detection.
      # vis_util.visualize_boxes_and_labels_on_image_array(
      #     image_np,
      #     np.squeeze(boxes),
      #     np.squeeze(classes).astype(np.int32),
      #     np.squeeze(scores),
      #     category_index,
      #     use_normalized_coordinates=True,
      #     line_thickness=8)

      image_np = cv2.resize(image_np, (800,600))
      image_np = cv2.drawMarker(image_np, (int(800 * norm_target[0]), int(600 * norm_target[1])), 200, cv2.MARKER_CROSS)
      cv2.imshow('object detection', image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break