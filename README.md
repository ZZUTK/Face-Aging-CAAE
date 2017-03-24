# Age Progression/Regression by Conditional Adversarial Autoencoder (CAAE)

TensorFlow implementation of the algorithm in the paper [Age Progression/Regression by Conditional Adversarial Autoencoder](http://web.eecs.utk.edu/~zzhang61/docs/papers/2017_CVPR_Age.pdf).

<p align="center">
  <img src="demo/method.png" width="500">
</p>


## Pre-requisites
* Python 2.7x
* Scipy
* TensorFlow
* [Training dataset (Aligend&Cropped Faces)](https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=sharing)

## Prepare the training dataset
You may used any dataset with labels of age and gender. In this demo, we use the [UTKFace](https://susanqq.github.io/UTKFace/) dataset. It is better to use algined and cropped faces. Please save and unzip `UTKFace.tar.gz` to the folder `data`. 

## Training
```
$ python main.py
```

The training process has been tested on NVIDIA TITAN X (12GB). The training time of 50 epochs on UTKFace (23,708 images in the size of 128x128x3) is about two and a half hours.

During the training, a new folder named `save` will be created, including four sub-folders: `summary`, `samples`, `test`, and `checkpoint`.

* `samples` saves the reconstructed faces at each epoch.
* `test` saves the testing results at each epoch (generated faces at different ages based on input faces).
* `checkpoint` saves the model.
* `summary` saves the batch-wise losses and intermediate outputs. To visualize the summary, 
```
$ cd save/summary
$ tensorboard --logdir .
```

After the training, you can check the folders `samples` and `test` to visulize the reconstruction and testing performance, respectively. The following shows the reconstruction (left) and testing (right) results. The first row in the reconstruction results (left) are testing samples that yield the testing results (right) in age ascending order from top to bottom.

<p align="center">
  <img src="demo/sample.png" width="430"> &nbsp <img src="demo/test.png" width="430">
</p>

The reconstruction loss vs. epoch is shown as follow, whose original record is saved in the folder `summary`.

<p align="center">
  <img src="demo/loss_epoch.jpg" width="500">
</p>




## Testing
```
$ python main.py --is_train False
```



## A demo of training process

The first row shows the input faces of different ages, and the other rows show the improvement of the output faces at each two epoch. From top to bottom, the output faces are in the age ascending order. 

<p align="center">
  <img src="demo/demo_train.gif" width="800">
</p>

<img src="demo/demo_train.gif">


## Files
* [`FaceAging.py`](FaceAging.py) is a class that builds and initialize the model, and implements training and testing related stuff
* [`ops.py`](ops.py) consists of functions called by `FaceAging.py` to implement options of convolution, deconvolution, fully connetion, leaky ReLU, load and save images.   
* [`main.py`](main.py) demonstrates `FaceAging.py`.
    
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
