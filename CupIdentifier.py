import os
HOME = os.getcwd()

!git clone https://github.com/SkalskiP/yolov9.git
%cd yolov9

!pip install -r requirements.txt -q
!pip install -q roboflow

import roboflow
from IPython.display import Image

!mkdir -p {HOME}/weights

!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt

!ls -la {HOME}/weights

!mkdir -p {HOME}/data

!python detect.py --weights {HOME}/weights/gelan-c.pt --conf 0.1 --source /content/Screenshot.png --device 0

%cd {HOME}/yolov9


from roboflow import Roboflow
rf = Roboflow(api_key="Owb7wL0INQDuAuAz9gth")
project = rf.workspace("nicolai-hoirup-nielsen").project("cup-detection-v2")
dataset = project.version(3).download("yolov9")

!python train.py \
--batch 8 --epochs 15 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data {dataset.location}/data.yaml \
--weights {HOME}/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml

!ls {HOME}/yolov9/runs/train/exp/
Image(filename=f"{HOME}/yolov9/runs/train/exp/results.png", width=700)
Image(filename=f"{HOME}/yolov9/runs/train/exp/confusion_matrix.png", width=700)
Image(filename=f"{HOME}/yolov9/runs/train/exp/val_batch0_pred.jpg", width=700)

%cd {HOME}/yolov9

!python val.py \
--img 640 --batch 8 --conf 0.001 --iou 0.7 --device 0 \
--data {dataset.location}/data.yaml \
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt

!python detect.py \
--img 640 --conf 0.1 --device 0 \
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt \
--source {dataset.location}/valid/images

import glob

for image_path in glob.glob(f'{HOME}/yolov9/runs/detect/exp7/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")
