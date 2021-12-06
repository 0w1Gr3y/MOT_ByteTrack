# Multiple Object Tracker with YOLOX and ByteTrack 
The source code in the folder "tracker" is copied from [ByteTrack repository](https://github.com/ifzhang/ByteTrack)
The folder "tracker" contains only the code required to run BYTETracker class and nothing else.

For the original implementation please visit [https://github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack).

Using this repository you can use YOLOX cloned from [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX) 
## Installation

### Prepare requirements
1. Install python requirements
```shell
pip3 install -r tracker/requirements.txt
```
2. Install pytorch [follow this manual](https://pytorch.org/get-started/locally/)
```shell
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

3. Install **pycocotools**
* For Ubuntu:
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
* For Windows:
```shell
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

5. Install **cython_box**
* Ubuntu
```shell
pip3 install cython_bbox
```
* Windows
```shell
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
```

### Install YOLOX
```shell
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX python setup.py develop
cd ..
```
download YOLOX weights to _weights_ folder from [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX)
#### Standard Models (source: [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX).
|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8| [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.7 | 48.0 | 11.1 |63.7 | 185.3 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |
#### Light Models (source: [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX).
|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/nano.py) |416  |25.8  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) |

### Prepare videos for input
You can download [this](https://www.youtube.com/watch?v=MNn9qKG2UFI&t=8s&ab_channel=KarolMajek) 4K traffic camera video from youtube

```shell
pip3 install youtube-dl
youtube-dl -f 313 MNn9qKG2UFI
```~~
