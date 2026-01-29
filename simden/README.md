# The Reproduction of SimDen Module

This tutorial will guide you on how to reproduce the dense radar point clouds by SimDen module.

# TODO: Tidy up the code of SimDen module following the roles like 'only saving points','only visualization' 

## Installation

1. Install the enviroments of YOLOv8 by referring the official website [YOLOv8](https://github.com/Pertical/YOLOv8). 
- The version of PyTorch and TorchVision we use are 2.0.0+cu118 and 0.15.1+cu118, which may be referred by you when the error of version mismatch appears. 
- OpenCV is optional but needed by demo and visualization.
2. Download the weight of YOLOv8-seg from [Baidu](https://pan.baidu.com/s/1D2EZ3Iwer9wKgrqFXPvzZA?pwd=st3f). And place it at `./weights/`

## Generate dense radar point clouds
You can generate the code with the following command.
```
cd ultralytics/examples/YOLOv8-Segmentation-ONNXRuntime-Python/

python main1.py --model /root/ultralytics/weight/yolov8l-seg.onnx --source /root/autodl-tmp/04853.jpg --conf 0.25
```