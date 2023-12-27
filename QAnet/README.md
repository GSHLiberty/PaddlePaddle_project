# QANet
A PaddlePaddle implementation of Google's [QANet](https://openreview.net/pdf?id=B14TlG-RW) (previously Fast Reading Comprehension (FRC)) from [ICLR2018](https://openreview.net/forum?id=B14TlG-RW). 

## Dataset
The dataset used for this task is [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/).
Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from common crawl with 840B tokens used for words.

## Requirements
  * Python>=2.7
  * NumPy
  * tqdm
  * paddlepaddle-gpu==0.0.0.post112
  * spacy==2.0.9

## Usage
To download and preprocess the data, run

```bash
# download SQuAD and Glove
sh download.sh
# preprocess the data, follwing https://github.com/andy840314/QANet-pytorch-
python preproc.py
```

### Run 
To train model with cuda:

```
python3 QANet_main.py --batch_size 32 --epochs 30 --with_cuda --use_ema 
```

To debug with small batches data:

```
python3 QANet_main.py --batch_size 32 --epochs 3 --with_cuda --use_ema --debug
```

### Pretrained Model
Pretrained model weights can be downloaded from [BaiduYun](https://pan.baidu.com/s/1c45vQTS3WqDJ7_Pio1JSMg) , keyword is 7kyz.

