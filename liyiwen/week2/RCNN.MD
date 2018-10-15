# 笔记: 目标检测之Faster RCNN系列

## 1. R-CNN基本框架

目标检测(Object detection)是计算机视觉领域的一个非常重要的研究问题。
相比于目标识别(Object classification)，目标检测还需要给出目标在图片中的位置。
一般来说，完成目标检测有两个大类的方法，比如one-stage方法，比如经典的YOLO系列算法，另一类就是two-stage的RCNN系列算法了。
目前也有一些综合了两类有点的算法。这篇笔记主要讲解RCNN系列的算法。

要找出目标的区域，一种蛮力算法是滑动窗口的方法。
对窗口位置和大小进行枚举，对每一个小块提取特征并分类，最后选择一个合适的位置。
这种做法显然无意义的计算过多，计算效率很低。

因此，R-CNN采用一种基于候选区域(Region proposal)的方法，大大降低需要考虑的区域，从而提高计算速度。

![RCNN](res/R-CNN.png)

对于一张图片，显然不是所有的部分都需要考虑，我们可以按照某种规则生成一些最有可能有目标的区域，叫做Regions of Interest(RoI)，在这个选取的过程中叫做Region proposals。
这些候选区域的形状是任意的，因此需要经过变形，变成CNN的输入尺寸。经过CNN提取出features map。
最后，对每一个种类，使用一个简单的二分类器（比如SVM），计算该RoI的类别。

<br />

我们分别来考察这些过程的细节。

### (1) Region proposals

现在有很多种方法可以生成类别独立的Region proposal，我们详细讲解一种叫做Selective search的算法。下图是Selective search的效果。

![selective-search](res/selective-search.png)

Selective search是基于区域合并的的算法，可以看成一种特殊的层次聚类。

首先使用一种基于图的图像分割算法[3]，得到初始区域的集合R。对R中的每一对相邻区域，计算这对区域的相似度，得到相似度的集合S.
不断的从S中取出最大的相似度，和对应的区域，将其合并，并将对应区域原本的相邻相似度从S中去除，同时将新的区域和其相邻相似度加入R和S。
当S为空时，此时的R即为最后的region proposals。

可以注意到，R中包含了所有算法中出现的区域，包含了不同的层次。关于更详细的算法描述可以参考下图。

![ss-algo](res/ss-algo.png)

<br />

### (2) 区域拉伸

候选区域的大小和宽高比并不固定，因此在送到卷积神经网络之前，需要拉伸到一个固定的大小。这个过程我们也有一些不同的方法可以选择。

![transform](res/transform-methods.png)

上图给出了四个例子，每个例子分成上下两行以及B,C,D三列。

B列对应的方法是，将原本的候选区域的较短边，稍微往外扩展，然后等比例拉伸到合适大小，对空白区域填充候选区域的平均值。C列是等比拉伸，然后填充。D列是改变宽高比，直接拉伸到合适大小。

除此之外，拉伸前还可以设置一个参数p，用来指定padding的像素数。此时候选区域会先多获取p个像素的上下文padding，然后再运用上述的拉伸方法。

### (3) 特征提取

特征提取的网络结构其实可以是任意的，包括VGGNet，ResNet都可以应用到这一过程，原始论文使用的是AlexNet，结构如下图。

![network](res/RCNN-network.png)

原始图像是3通道的224x224的图象。第一个卷积层的卷积核的尺寸是5x5x48，

## 2. SPP神经网络


## 3. fast R-CNN


## 4. faster R-CNN


-------

1. [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)
2. [Selective Search for Object Recognition](https://www.koen.me/research/pub/uijlings-ijcv2013-draft.pdf)
3. [Efficient Graph-Based Image Segmentation](http://cs.brown.edu/people/pfelzens/segment/)
4. [ImageNet Classification with Deep Convolutional
Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)