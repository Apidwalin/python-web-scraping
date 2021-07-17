# Intro to Object Detection Colab

Welcome to the object detection colab! This demo will take you through the steps of running an "out-of-the-box" detection model in SavedModel format on a collection of images.



Imports


```python
!pip install -U --pre tensorflow=="2.2.0"
```

    Collecting tensorflow==2.2.0
      Downloading tensorflow-2.2.0-cp37-cp37m-win_amd64.whl (459.2 MB)
    Collecting h5py<2.11.0,>=2.10.0
      Downloading h5py-2.10.0-cp37-cp37m-win_amd64.whl (2.5 MB)
    Collecting gast==0.3.3
      Downloading gast-0.3.3-py2.py3-none-any.whl (9.7 kB)
    Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (3.3.0)
    Collecting tensorboard<2.3.0,>=2.2.0
      Downloading tensorboard-2.2.2-py3-none-any.whl (3.0 MB)
    Requirement already satisfied: numpy<2.0,>=1.16.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (1.18.1)
    Requirement already satisfied: absl-py>=0.7.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (0.13.0)
    Requirement already satisfied: google-pasta>=0.1.8 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (0.2.0)
    Requirement already satisfied: protobuf>=3.8.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (3.17.3)
    Collecting scipy==1.4.1
      Downloading scipy-1.4.1-cp37-cp37m-win_amd64.whl (30.9 MB)
    Requirement already satisfied: grpcio>=1.8.6 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (1.34.1)
    Requirement already satisfied: astunparse==1.6.3 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (1.6.3)
    Collecting tensorflow-estimator<2.3.0,>=2.2.0
      Downloading tensorflow_estimator-2.2.0-py2.py3-none-any.whl (454 kB)
    Requirement already satisfied: wrapt>=1.11.1 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (1.12.1)
    Requirement already satisfied: termcolor>=1.1.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (1.1.0)
    Requirement already satisfied: six>=1.12.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (1.14.0)
    Requirement already satisfied: wheel>=0.26 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (0.36.2)
    Requirement already satisfied: keras-preprocessing>=1.1.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorflow==2.2.0) (1.1.2)
    Requirement already satisfied: google-auth<2,>=1.6.3 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (1.33.0)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (3.2)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (0.4.4)
    Requirement already satisfied: setuptools>=41.0.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (57.2.0)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (2.24.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (1.8.0)
    Requirement already satisfied: werkzeug>=0.11.15 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (2.0.1)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (4.2.2)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (4.7.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (0.2.8)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (1.3.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (0.4.8)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (1.25.11)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (2020.6.20)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\apidwalin\anaconda3\envs\tensorflow1\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow==2.2.0) (3.1.1)
    Installing collected packages: tensorflow-estimator, tensorboard, scipy, h5py, gast, tensorflow
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 2.5.0
        Uninstalling tensorflow-estimator-2.5.0:
          Successfully uninstalled tensorflow-estimator-2.5.0
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 2.5.0
        Uninstalling tensorboard-2.5.0:
          Successfully uninstalled tensorboard-2.5.0
      Attempting uninstall: scipy
        Found existing installation: scipy 1.7.0
        Uninstalling scipy-1.7.0:
          Successfully uninstalled scipy-1.7.0
      Attempting uninstall: h5py
        Found existing installation: h5py 3.1.0
        Uninstalling h5py-3.1.0:
          Successfully uninstalled h5py-3.1.0
      Attempting uninstall: gast
        Found existing installation: gast 0.4.0
        Uninstalling gast-0.4.0:
          Successfully uninstalled gast-0.4.0
    Successfully installed gast-0.3.3 h5py-2.10.0 scipy-1.4.1 tensorboard-2.2.2 tensorflow-2.2.0 tensorflow-estimator-2.2.0
    

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow-gpu 2.5.0 requires gast==0.4.0, but you have gast 0.3.3 which is incompatible.
    tensorflow-gpu 2.5.0 requires h5py~=3.1.0, but you have h5py 2.10.0 which is incompatible.
    tensorflow-gpu 2.5.0 requires numpy~=1.19.2, but you have numpy 1.18.1 which is incompatible.
    tensorflow-gpu 2.5.0 requires six~=1.15.0, but you have six 1.14.0 which is incompatible.
    tensorflow-gpu 2.5.0 requires tensorboard~=2.5, but you have tensorboard 2.2.2 which is incompatible.
    tensorflow-gpu 2.5.0 requires tensorflow-estimator<2.6.0,>=2.5.0rc0, but you have tensorflow-estimator 2.2.0 which is incompatible.
    


```python
import os
import pathlib

# Clone the tensorflow models repository if it doesn't already exist
if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  !git clone --depth 1 https://github.com/tensorflow/models
```


```python
# Install the Object Detection API
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```


      File "<ipython-input-3-1cb5b0736a81>", line 3
        cd models/research/
                ^
    SyntaxError: invalid syntax
    



```python
import io
import os
import scipy.misc
import numpy as np
import six
import time

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

%matplotlib inline
```


```python
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Load the COCO Label Map
category_index = {
    1: {'id': 1, 'name': 'person'},
    2: {'id': 2, 'name': 'bicycle'},
    3: {'id': 3, 'name': 'car'},
    4: {'id': 4, 'name': 'motorcycle'},
    5: {'id': 5, 'name': 'airplane'},
    6: {'id': 6, 'name': 'bus'},
    7: {'id': 7, 'name': 'train'},
    8: {'id': 8, 'name': 'truck'},
    9: {'id': 9, 'name': 'boat'},
    10: {'id': 10, 'name': 'traffic light'},
    11: {'id': 11, 'name': 'fire hydrant'},
    13: {'id': 13, 'name': 'stop sign'},
    14: {'id': 14, 'name': 'parking meter'},
    15: {'id': 15, 'name': 'bench'},
    16: {'id': 16, 'name': 'bird'},
    17: {'id': 17, 'name': 'cat'},
    18: {'id': 18, 'name': 'dog'},
    19: {'id': 19, 'name': 'horse'},
    20: {'id': 20, 'name': 'sheep'},
    21: {'id': 21, 'name': 'cow'},
    22: {'id': 22, 'name': 'elephant'},
    23: {'id': 23, 'name': 'bear'},
    24: {'id': 24, 'name': 'zebra'},
    25: {'id': 25, 'name': 'giraffe'},
    27: {'id': 27, 'name': 'backpack'},
    28: {'id': 28, 'name': 'umbrella'},
    31: {'id': 31, 'name': 'handbag'},
    32: {'id': 32, 'name': 'tie'},
    33: {'id': 33, 'name': 'suitcase'},
    34: {'id': 34, 'name': 'frisbee'},
    35: {'id': 35, 'name': 'skis'},
    36: {'id': 36, 'name': 'snowboard'},
    37: {'id': 37, 'name': 'sports ball'},
    38: {'id': 38, 'name': 'kite'},
    39: {'id': 39, 'name': 'baseball bat'},
    40: {'id': 40, 'name': 'baseball glove'},
    41: {'id': 41, 'name': 'skateboard'},
    42: {'id': 42, 'name': 'surfboard'},
    43: {'id': 43, 'name': 'tennis racket'},
    44: {'id': 44, 'name': 'bottle'},
    46: {'id': 46, 'name': 'wine glass'},
    47: {'id': 47, 'name': 'cup'},
    48: {'id': 48, 'name': 'fork'},
    49: {'id': 49, 'name': 'knife'},
    50: {'id': 50, 'name': 'spoon'},
    51: {'id': 51, 'name': 'bowl'},
    52: {'id': 52, 'name': 'banana'},
    53: {'id': 53, 'name': 'apple'},
    54: {'id': 54, 'name': 'sandwich'},
    55: {'id': 55, 'name': 'orange'},
    56: {'id': 56, 'name': 'broccoli'},
    57: {'id': 57, 'name': 'carrot'},
    58: {'id': 58, 'name': 'hot dog'},
    59: {'id': 59, 'name': 'pizza'},
    60: {'id': 60, 'name': 'donut'},
    61: {'id': 61, 'name': 'cake'},
    62: {'id': 62, 'name': 'chair'},
    63: {'id': 63, 'name': 'couch'},
    64: {'id': 64, 'name': 'potted plant'},
    65: {'id': 65, 'name': 'bed'},
    67: {'id': 67, 'name': 'dining table'},
    70: {'id': 70, 'name': 'toilet'},
    72: {'id': 72, 'name': 'tv'},
    73: {'id': 73, 'name': 'laptop'},
    74: {'id': 74, 'name': 'mouse'},
    75: {'id': 75, 'name': 'remote'},
    76: {'id': 76, 'name': 'keyboard'},
    77: {'id': 77, 'name': 'cell phone'},
    78: {'id': 78, 'name': 'microwave'},
    79: {'id': 79, 'name': 'oven'},
    80: {'id': 80, 'name': 'toaster'},
    81: {'id': 81, 'name': 'sink'},
    82: {'id': 82, 'name': 'refrigerator'},
    84: {'id': 84, 'name': 'book'},
    85: {'id': 85, 'name': 'clock'},
    86: {'id': 86, 'name': 'vase'},
    87: {'id': 87, 'name': 'scissors'},
    88: {'id': 88, 'name': 'teddy bear'},
    89: {'id': 89, 'name': 'hair drier'},
    90: {'id': 90, 'name': 'toothbrush'},
}
```


```python
# Download the saved model and put it into models/research/object_detection/test_data/
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz
!tar -xf efficientdet_d5_coco17_tpu-32.tar.gz
!mv efficientdet_d5_coco17_tpu-32/ models/research/object_detection/test_data/
```


```python
start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('models/research/object_detection/test_data/efficientdet_d5_coco17_tpu-32/saved_model/')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')
```


```python
import time

image_dir = 'models/research/object_detection/test_images'

elapsed = []
for i in range(2):
  image_path = os.path.join(image_dir, 'image' + str(i + 1) + '.jpg')
  image_np = load_image_into_numpy_array(image_path)
  input_tensor = np.expand_dims(image_np, 0)
  start_time = time.time()
  detections = detect_fn(input_tensor)
  end_time = time.time()
  elapsed.append(end_time - start_time)

  plt.rcParams['figure.figsize'] = [42, 21]
  label_id_offset = 1
  image_np_with_detections = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)
  plt.subplot(2, 1, i+1)
  plt.imshow(image_np_with_detections)

mean_elapsed = sum(elapsed) / float(len(elapsed))
print('Elapsed time: ' + str(mean_elapsed) + ' second per image')
```
