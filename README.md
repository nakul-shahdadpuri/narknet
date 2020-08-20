# Narknet [Under development]
Object Detection for Computer Vision using YOLOv3.

This repository a computer vision library , using YOLOv3 machine learning model. The program is implemented in python3 and will be converted to cython in due time.

## Dependencies:
1. **Python --3.7.6**
2. **Opencv --4.2.0**
3. **Axel --2.17.5**
4. **Conda --4.8.3**
5. **Numpy --1.18.1**
6. **Requests --2.23.0**

## Setup:
1. Set up an anaconda environment, 'https://docs.anaconda.com/anaconda/install/linux/'
2. sudo apt install axel (For debian based distros)
3. After set up of conda environment, install opencv2 and pandas.
4. **conda install -c menpo opencv (For opencv)**
5. **conda install pandas (For Pandas)**

## Installation

```sh
git clone https://github.com/nakul-shahdadpuri/narknet.git
cd narknet/
cd Weights/
chmod u+x GetWeights.sh
./GetWeights.sh
```

## Running narknet

## Resources
1. Non Max Suppression 'https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c'
2. YOLOv3 model 'https://pjreddie.com/darknet/yolo/'
3. cv2.BlobFromImage 'https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/'
4. OpenCv Documentation 'https://docs.opencv.org/2.4/'
5. DeepSort Repo 'https://github.com/nwojke/deep_sort' 
6. SORT Paper 'https://arxiv.org/abs/1602.00763'
7. Deep Sort 'https://medium.com/analytics-vidhya/yolo-v3-real-time-object-tracking-with-deep-sort-4cb1294c127f'
