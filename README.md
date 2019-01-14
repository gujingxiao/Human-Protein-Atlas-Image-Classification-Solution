# Human-Protein-Atlas-Image-Classification-Solution
Silver Metal Solution for [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification) held by Kaggle

For my previous solutions, Keras or Tensorflow is my first choose. But later I found most top kagglers use Pytorch or FastAi. So for this challenge, I struggled to learn Pytorch and FastAi. I have to say that my code is highly inspired by [omallo's starter code](https://github.com/omallo/kaggle-hpa-fastai) (But I don't know why omallo gives up the challenge)

I joined the competetion pretty late (about 4 weeks ago) and only have two 1070Ti(16 GB). So I hope my simple solution can help those who don't have powerful devices. Simple models can also get good results.

## Instructions
All configs are in train.py. Just put the correct path and modify some other parameters. Pytorch 1.0 and FastAi 1.0.37 are required.

#### Single Models
|Models|Image Size|Batch Size|Loss Function|Local CV|Public LB|Private LB|
|:---|:---|:---|:---|:---|:---|:---|
|densenet121|512 x 512|32|Weighted Focal Loss with F1|0.577|0.527|0.509|
|resnet34|512 x 512|64|Weighted Focal Loss with F1|0.581|0.517|0.502|
|resnet50|512 x 512|32|Weighted Focal Loss with F1|0.573|0.509|0.497|
|densenet169|512 x 512|16|Weighted Focal Loss with F1|0.565|0.509|0.493|
|resnet101|512 x 512|16|Weighted Focal Loss with F1|0.580|0.498|0.495|
|resnet18|512 x 512|96|Weighted Focal Loss with F1|0.588|0.496|0.487|
|resnet101|256 x 256|64|Weighted Focal Loss with F1|0.552|0.488|0.490|
|resnet50|256 x 256|128|Weighted Focal Loss with F1|0.561|0.483|0.473|
|densenet121|256 x 256|128|Weighted Focal Loss with F1|0.562|0.475|0.461|

#### Ensemble Models
After three models ensemble, the submission still has 407 blank predictions and 23 predictions having more than 5 results. In the final submission, I deal with this situation and use Leak 259. Well, the final submission still has 5 blank predictions but I have no chance to try, haha.

|Models|Local CV|Public LB|Private LB|
|:---|:---|:---|:---|
|dns121 + rsn34 + rsn50|0.613|0.536|0.515|
|Final Submission|0.636|0.602|0.525|

#### About Pretrained Models & External Data
Like many top kagglers say, external data is not a neccesity for high rank but it does help. I download 50000+ images from [HPAv18](http://v18.proteinatlas.org/). Pretrained models are all from [open resources](https://github.com/pytorch/vision)
