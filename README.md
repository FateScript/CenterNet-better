# CenterNet

This repo is implemented based on my dl_lib, some parts of  code in my dl_lib  is based on [detectron2](https://github.com/facebookresearch/detectron2).

## Motivation

[Objects as Points](https://arxiv.org/abs/1904.07850) is one of my favorite paper in object detection area. However, its [code](https://github.com/xingyizhou/CenterNet/blob/master/README.md) is a little difficult to understand. I believe that CenterNet could get higher pts and implemented in a more elegant way, so I write this repo.

## Performance

This repo use less training time to get a better performance, it nearly spend half training time and get 1~2 pts higher mAP compared with the old repo. Here is the table of performance.

| Backbone     |  mAP    |  FPS    |  V100 FPS |  trained model    |  
|:------------:|:-------:|:-------:|:---------:|:-----------------:|  
|ResNet-18     | 29.8    | 92      | 113       | [google drive](https://drive.google.com/open?id=1D3tO95sdlsh9egOjOg0N-2HHmMfqbt5X)   |  
|ResNet-50     | 34.9    | 57      | 71        | [google drive](https://drive.google.com/open?id=1t5Bw520_fJrn3aeSVxDBYNIgwpNdLR5s)   |  
|ResNet-101    | 36.8    | 43      | 50        | [google drive](https://drive.google.com/open?id=1762Y93i9QreUTHq-87Ir73R2nNcrHuk0)   |  

## What\'s New?
* **treat config as a object.** You could run your config file and check the config value, which is really helpful for debug.
* **Common training / testing scripts in default.** you just need to invoke `dl_train/test --num-gpus x` in your playground and your projects only need to include all project-specific configs and network modules.
* **Performance report is dumped automaticly.** After your training is over, we will evaluate your model automatically and generate a markdown file.
* **Vectorize some operations.** This improves the speed and efficiency.

## What\'s comming
  - [ ] Support DLA backbone
  - [ ] Support Hourglass backbone
  - [ ] Support KeyPoints dataset

## Get started
### Requirements
* Python >= 3.6
* PyTorch >= 1.3
* torchvision that matches the PyTorch installation.
* OpenCV
* pycocotools
```shell
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
* GCC >= 4.9
```shell
gcc --version
```

### Installation

Make sure that your get at least one gpu when compiled. Run:
```shell
pip install -e .
```

### Training
For example, if you want to train CenterNet with resnet-18 backbone, run:
```shell
cd playground/centernet.res18.coco.512size
dl_train --num-gpus 8
```
After training process, a README.md file will be generated automatically and this file will report your model\'s performance.  

NOTE: For ResNet-18 and ResNet-50 backbone, we suppose your machine has over 150GB Memory for training. If your memory is not enough, please change NUM_WORKER (in config.py) to a smaller value.

### Testing and Evaluation
```shell
dl_test --num-gpus 8 
```
test downloaded model:
```shell
dl_test --num-gpus 8  MODEL.WEIGHTS path/to/your/save_dir/ckpt.pth 
```

## Acknowledgement
* [detectron2](https://github.com/facebookresearch/detectron2)
* [CenterNet](https://github.com/xingyizhou/CenterNet)

## Coding style

please refer to  [google python coding style](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)

## Citing CenterNet-better

If you use CenterNet-better in your research or wish to refer to the baseline results published in this repo, please use the following BibTeX entry.

```BibTeX
@misc{wang2020centernet_better,
  author =       {Feng Wang},
  title =        {CenterNet-better},
  howpublished = {\url{https://github.com/FateScript/CenterNet-better}},
  year =         {2020}
}
```
