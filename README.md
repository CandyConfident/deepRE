


# 复现CNN 和 lstm+att 做关系抽取

##实验结果
### cnn
Micro-averaged result (excluding Other):
P = 2004/2262 =  84.36%     R = 2004/2449 =  79.23.%     F1 =  81.18%

MACRO-averaged result (excluding Other):
P =  83.66%		R =  77.15%	F1 =  80.72%

### a-cnn
#### 此结果运行代码 https://github.com/lawlietAi/pytorch-acnn-model  所得
Micro-averaged result (excluding Other):
P = 2004/2262 =  85.59%     R = 2004/2449 =  79.83%     F1 =  82.78%

MACRO-averaged result (excluding Other):
P =  85.66%		R =  78.15%	F1 =  81.72%

### lstm+att
Micro-averaged result (excluding Other):
P = 2004/2262 =  83.59%     R = 2004/2449 =  79.83%     F1 =  81.75%

MACRO-averaged result (excluding Other):
P =  82.66%		R =  79.15%	F1 =  80.72%

# Usage
## Train
#! /bin/bash

    mkdir -p saved_models
    
    CUDA_VISIBLE_DEVICES=2 python3  src/train.py  --num_epochs=200 --word_dim=50
    python src/train.py  --num_epochs=200 --word_dim=50 --test
    src/scorer.pl data/results.txt data/test_keys.txt 
    
    python src/train.py  --num_epochs=200 --word_dim=300
    python src/train.py  --num_epochs=200 --word_dim=300 --test

# Data
SemEval-2010 Task #8

# Reference

- Relation Classification via Convolutional Deep Neural Network (COLING 2014), D Zeng et al. 
- Relation Extraction: Perspective from Convolutional Neural Networks (NAACL 2015), TH Nguyen et al. 
- Zhou. (2016). Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification. ACL
- https://github.com/lawlietAi/pytorch-acnn-model
- https://github.com/FrankWork/conv_relation
