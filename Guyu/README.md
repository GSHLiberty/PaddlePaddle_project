# Guyu (谷雨)
A PaddlePaddle implementation of pre-training and fine-tuning framework for text generation.

Backbone code for "An Empirical Investigation of Pre-Trained Transformer Language Models for Open-Domain Dialogue Generation": https://arxiv.org/abs/2003.04195
#### Pre-training:

```
./prepare_data.sh
./train.sh
./inference.sh
```

#### Fine-tuning
Example: chat-bot

```
cd chat_bot
./prepare_data.sh
./fine_tune.sh
./inference.sh
```

#### Web Api:
```
./deploy.sh
```

#### Pre-trained models
Pretrained model weights can be downloaded from [BaiduYun](https://pan.baidu.com/s/1KwAkwOqQqCX6X2HMOxDkeA) , keyword is 6ku6. You can download it under model/.

