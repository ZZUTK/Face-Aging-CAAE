# Age Progression/Regression by Conditional Adversarial Autoencoder

TensorFlow implementation of the paper [Age Progression/Regression by Conditional Adversarial Autoencoder](http://web.eecs.utk.edu/~zzhang61/docs/papers/2017_CVPR_Age.pdf).

<img src="demo/method.png" width="500">


## Pre-requisites
* Python 2.7x
* Scipy
* TensorFlow
* [Training dataset (Aligend&Cropped Faces)](https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=sharing)

## Prepare the training dataset
Download a training dataset with labels of age and gender. We use the [UTKFace](https://susanqq.github.io/UTKFace/) dataset in the demo. It is better to use algined and cropped faces. 

Please save and unzip ```UTKFace.tar.gz``` to the folder ```data```. 

## Training
```
$ python MAIN.py
```

The training process has been tested on NVIDIA TITAN X (12GB). The training time of 50 epochs on UTKFace (23,708 images in the size of 128x128x3) is about two and a half hours.

## Testing
```
$ python MAIN.py --is_train False
```



## A demo of training process
<img src="https://github.com/ZZUTK/Age-Progression/blob/master/demo/demo_train.gif">


## Files
* [`FaceProgression.py`](https://github.com/ZZUTK/Age-Progression/blob/master/FaceProgression.py) is a class that builds and initialize the model, and implements training and testing related stuff
    * [`ops.py`](https://github.com/ZZUTK/Age-Progression/blob/master/ops.py) consists of functions called by `FaceProgression.py` to implement options of convolution, deconvolution, fully connetion, leaky ReLU, batch normalization, load and save images, etc.
    
* [`MAIN.py`](https://github.com/ZZUTK/Age-Progression/blob/master/MAIN.py) demonstrates `FaceProgression.py`.
    * [`checkGPU.py`](https://github.com/ZZUTK/Age-Progression/blob/master/checkGPU.py) is called in `MAIN.py` to check GPU status, ensuring enough momery (defualt 3.2GB).
    
* [`run_lis.sh`](https://github.com/ZZUTK/Age-Progression/blob/master/run_list.sh) run a list of tests via bash 

## Citation
Zhifei Zhang, Yang Song, and Hairong Qi. "Age Progression/Regression by Conditional Adversarial Autoencoder." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.
```
@inproceedings{zhang2017age,
  title={Age Progression/Regression by Conditional Adversarial Autoencoder},
  author={Zhang, Zhifei and Song, Yang and Qi, Hairong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```
