# Keras EfficientDet

Keras implementation of EfficientDet object detection as described in EfficientDet: Scalable and Efficient Object Detection by Mingxing Tan, Ruoming Pang, Quoc V. Le.

This code is borrowed from Keras Implementation of this model at https://github.com/fizyr/keras-retinanet and updated to run on Stanford Drone Data Set


# Installation

1. Clone this repository.
2. Ensure numpy is installed using pip install numpy --user
3. In the repository, execute pip install . --user. Note that due to inconsistencies with how tensorflow should be installed, this package does not define a dependency on tensorflow as it will try to install that (which at least on Arch Linux results in an incorrect installation). Please make sure tensorflow is installed as per your systems requirements.
4. Alternatively, you can run the code directly from the cloned repository, however you need to run python setup.py build_ext --inplace to compile Cython code first.
5. Optionally, install pycocotools if you want to train / test on the MS COCO dataset by running pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI.


# Running directly from the repository:

For training on a [custom dataset], a CSV file can be used as a way to pass the data. See below for more details on the format of these CSV files.

To train using your CSV, run:
Running directly from the repository:

`!python train.py --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 --weighted-bifpn --epochs 25 --random-transform --compute-val-loss --batch-size 16 --steps 209 csv /file_path/train.csv /file_path/labels.csv --val /file_path/val.csv`
