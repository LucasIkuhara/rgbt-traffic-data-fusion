# Dataset and Experiment

The dataset used in this project is called AAU RainSnow and it is comprised of 130,800 pairs of images taken from traffic monitoring cameras spread across 7 different locations in Denmark.
Each pair contains one RGB image and one thermal grayscale image, both at 640x480 resolution. 
Most importantly, many of the images were taken in low-visibility scenarios at which object detection becomes significantly harder.
In the RGB images this often means the camera has drops of water splattered across its lens or the image was simply too dark whilist in the thermal images low-contrast is often the biggest challenge to precise OD.
However, it is important to note that this instances of low image quality do not necessarily happen simultaneously. 
In many cases, objects might be difficult to spot from the RGB camera PoV but perfectly visible in its thermal counterpart and vice-versa.
Therefore, scenarios like this present a large opportunity for performance improvements leveraging the fusion of both sources into a combined detection.

In this work we use compare the performance achieved in object detection using YOLOv8-s in RGB iamges to the perfomance of YOLOv8 fine-tuned to OD in thermal images and their fused detections using different bounding-box combination techniques.
