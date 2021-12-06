# Multiple Object Tracker with YOLOX and ByteTrack 
The source code from the folder "tracker" is copied from [ByteTrack](https://github.com/ifzhang/ByteTrack)

# Installation

pip3 install -r tracker/requirements.txt
https://pytorch.org/get-started/locally/
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113
-f https://download.pytorch.org/whl/cu113/torch_stable.html
pycocotools
    Ubuntu
        pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    Windows
        pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
cython_box
    Ubuntu
        pip3 install cython_bbox
    Windows
        pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

# YOLOX

git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX python setup.py develop cd ..

# download weights to 'weights' folder
# prepare videos for input
    ! https://www.youtube.com/watch?v=MNn9qKG2UFI&t=8s&ab_channel=KarolMajek

# videos



https://www.youtube.com/watch?v=jjlBnrzSGjc&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&ab_channel=PanasonicSecurity
https://www.youtube.com/watch?v=MNn9qKG2UFI&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=2&ab_channel=KarolMajek
https://www.youtube.com/watch?v=PSf09R3D7Lo&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=3&ab_channel=PanasonicSecurity
https://www.youtube.com/watch?v=s5Bf_KEtoyM&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=9&ab_channel=PanasonicSecurity
https://www.youtube.com/watch?v=PJ5xXXcfuTc&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=40&ab_channel=Supercircuits


