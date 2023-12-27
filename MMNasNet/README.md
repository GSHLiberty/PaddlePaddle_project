# Deep Multimodal Neural Architecture Search (MMNasNet)

This repository corresponds to the PaddlePaddle implementation of the MMNasNet for VQA.

## Prerequisites

#### Software and Hardware Requirements

You may need a machine with at least **1 GPU (>= 8GB)**, **20GB memory** and **50GB free disk space**.  We strongly recommend to use a SSD drive to guarantee high-speed I/O.

You should first install some necessary packages.

1. Install [Python](https://www.python.org/downloads/) == 3.9.18
2. Install [Cuda](https://developer.nvidia.com/cuda-toolkit) == 11.2 and [cuDNN](https://developer.nvidia.com/cudnn)
3. Install paddlepaddle-gpu == 0.0.0.post112.
4. Install [SpaCy](https://spacy.io/) and initialize the [GloVe](https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz) as follows:

	```bash
	$ pip install -r requirements.txt
	$ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
	$ pip install en_vectors_web_lg-2.1.0.tar.gz
	```

#### Dataset Setup ——VQA-V2

- Image Features

 The image features are extracted using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) strategy, with each image being represented as an dynamic number (from 10 to 100) of 2048-D features. We store the features for each image in a `.npz` file. You can prepare the visual features by yourself or download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ?e=2BZl8O) or [BaiduYun](https://pan.baidu.com/s/1C7jIWgM3hFPv-YXJexItgw#list/path=%2F). The downloaded files contains three files: **train2014.tar.gz, val2014.tar.gz, and test2015.tar.gz**, corresponding to the features of the train/val/test images for *VQA-v2*, respectively. 

All the image feature files are unzipped and placed in the `data/vqa/feats` folder to form the following tree structure:

```angular2html
|-- data
	|-- vqa
	|  |-- feats
	|  |  |-- train2014
	|  |  |  |-- COCO_train2014_...jpg.npz
	|  |  |  |-- ...
	|  |  |-- val2014
	|  |  |  |-- COCO_val2014_...jpg.npz
	|  |  |  |-- ...
	|  |  |-- test2015
	|  |  |  |-- COCO_test2015_...jpg.npz
	|  |  |  |-- ...
```

- QA Annotations

Download all the annotation `json` files for VQA-v2, including the [train questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip), [train answers](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), and [val answers](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip). 

In addition, we use the VQA samples from the Visual Genome to augment the training samples. We pre-processed these samples by two rules: 

1. Select the QA pairs with the corresponding images appear in the MS-COCO *train* and *val* splits; 
2. Select the QA pairs with the answer appear in the processed answer list (occurs more than 8 times in whole *VQA-v2* answers).

We provide our processed vg questions and annotations files, you can download them from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EmVHVeGdck1IifPczGmXoaMBFiSvsegA6tf_PqxL3HXclw) or [BaiduYun](https://pan.baidu.com/s/1QCOtSxJGQA01DnhUg7FFtQ#list/path=%2F).

All the QA annotation files are unzipped and placed in the `data/vqa/raw` folder to form the following tree structure:


```angular2html
|-- data
	|-- vqa
	|  |-- raw
	|  |  |-- v2_OpenEnded_mscoco_train2014_questions.json
	|  |  |-- v2_OpenEnded_mscoco_val2014_questions.json
	|  |  |-- v2_OpenEnded_mscoco_test2015_questions.json
	|  |  |-- v2_OpenEnded_mscoco_test-dev2015_questions.json
	|  |  |-- v2_mscoco_train2014_annotations.json
	|  |  |-- v2_mscoco_val2014_annotations.json
	|  |  |-- VG_questions.json
	|  |  |-- VG_annotations.json
```


## Training

The following script will start training with the default hyperparameters:

```bash
$ python3 run.py --RUN='train' --MODEL='mmnasnet' --DATASET='vqa'
```
- ``--RUN={'train','val','test'}`` to set the mode to be executed.

All checkpoint files will be saved to:

```
ckpts/ckpt_<VERSION>/epoch<EPOCH_NUMBER>.pkl
```

and the training log file will be placed at:

```
results/log/log_run_<VERSION>.txt
```

To add：

1. ```--VERSION=str```, e.g.```--VERSION='small_model'``` to assign a name for your this model.

2. ```--GPU=str```, e.g.```--GPU='2'``` to train the model on specified GPU device.

3. ```--NW=int```, e.g.```--NW=8``` to accelerate I/O speed.

4. ```--MODEL={'small', 'large'}```  ( Warning: The large model will consume more GPU memory, maybe [Multi-GPU Training and Gradient Accumulation](#Multi-GPU-Training-and-Gradient-Accumulation) can help if you want to train the model with limited GPU memory.)

5. ```--SPLIT={'train', 'train+val', 'train+val+vg'}``` can combine the training datasets as you want. The default training split is ```'train+val+vg'```.  Setting ```--SPLIT='train'```  will trigger the evaluation script to run the validation score after every epoch automatically.

6. ```--RESUME=True``` to start training with saved checkpoint parameters. In this stage, you should assign the checkpoint version```--CKPT_V=str``` and the resumed epoch number ```CKPT_E=int```.

7. ```--MAX_EPOCH=int``` to stop training at a specified epoch number.

8. ```--PRELOAD=True``` to pre-load all the image features into memory during the initialization stage (Warning: needs extra 25~30GB memory and 30min loading time from an HDD drive).


####  Multi-GPU Training and Gradient Accumulation

We recommend to use the GPU with at least 8 GB memory, but if you don't have such device, don't worry, we provide two ways to deal with it:

1. _Multi-GPU Training_: 

    If you want to accelerate training or train the model on a device with limited GPU memory, you can use more than one GPUs:

	Add ```--GPU='0, 1, 2, 3...'```

    The batch size on each GPU will be adjusted to `BATCH_SIZE`/#GPUs automatically.

2. _Gradient Accumulation_: 

    If you only have one GPU less than 8GB, an alternative strategy is provided to use the gradient accumulation during training:
	
	Add ```--ACCU=n```  
	
    This makes the optimizer accumulate gradients for`n` small batches and update the model weights at once. It is worth noting that  `BATCH_SIZE` must be divided by ```n``` to run this mode correctly. 


## Validation and Testing

**Warning**: If you train the model use ```--MODEL``` args or multi-gpu training, it should be also set in evaluation.


#### Offline Evaluation

Offline evaluation only support the VQA 2.0 *val* split. If you want to evaluate on the VQA 2.0 *test-dev* or *test-std* split, please see [Online Evaluation](#Online-Evaluation).

There are two ways to start:

(Recommend)

```bash
$ python3 run.py --RUN='val' --CKPT_V=str --CKPT_E=int
```

or use the absolute path instead:

```bash
$ python3 run.py --RUN='val' --CKPT_PATH=str
```


#### Online Evaluation

The evaluations of both the VQA 2.0 *test-dev* and *test-std* splits are run as follows:

```bash
$ python3 run.py --RUN='test' --CKPT_V=str --CKPT_E=int
```

Result files are stored in ``results/result_test/result_run_<CKPT_V>_<CKPT_E>.json``

For VQA-v2, the result file is uploaded the [VQA challenge website](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to evaluate the scores on *test-dev* or *test-std* split.


## Pretrained models

We provide a pretrained model, namely the `small` model.  


The model can be downloaded from  [BaiduYun](https://pan.baidu.com/s/1-PXOFwybTV_4he6iNC0K6w),  keyword is : j84i. You should unzip and put them to the correct folders as follows:

```angular2html
|-- ckpt
	|-- ckpt_small
	|  |-- MMNasNet-small_model_full.pdparams
```

