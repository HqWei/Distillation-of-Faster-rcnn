# Distillation-of-Faster-rcnn
Distillation for faster rcnn in classification,regression,feature level,feature level +mask

## Detail in my csdn blog:
https://blog.csdn.net/qq_33547191/article/details/95014337
###
https://blog.csdn.net/qq_33547191/article/details/95049838

## The code is heavily borrowed from :

### 1.Distillation for faster rcnn in classification,regression,feature level
http://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf


### 2.Distillation for faster rcnn in feature level +mask

http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.pdf

#### code:
 https://github.com/twangnh/Distilling-Object-Detectors
 
 
 main.py 里面介绍在哪里加入蒸馏
 adaplayer.py 是针对teacher和student的feature map大小不相同时进行特征图大小进行变换
 loss.py是如何计算分类和回归的蒸馏loss，其实就是计算他们的相似度
 
 蒸馏在teacher和student的精度差距非常大时效果特别明显。
 
