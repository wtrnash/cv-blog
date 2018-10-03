#  Clustering 聚类



[TOC]

聚类是无监督学习中研究最多、应用最广的方法。



聚类算法可分为

1. 原型聚类

   - k-means 算法

   - 学习向量量化 (LVQ) 算法

   - 高斯混合聚类

2. 密度聚类

   - DBSCAN 算法

3. 层次聚类

   - AGNES 算法





## 1. 基本概念



### 1.1 聚类任务

聚类尝试将数据集中的样本划分为若干个通常是不想交的子集，每个子集称为一个“簇”，每个簇有其潜在的概念语义，算法事先不知道。



聚类任务用数学表达：假设样本集 $D = \left\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { m } \right\}$ 包含 $m$ 个无标记样本，每个样本 $\boldsymbol { x } _ { i } = \left( x _ { i 1 } ; x _ { i 2 } ; \ldots ; x _ { i n } \right)$ 是一个 $n$ 维特征向量，则聚类算法将样本集 $D$ 划分为 $k$ 个不相交的簇 $\left\{ C _ { l } | l = 1,2 ; \ldots , k \right\}$，其中 $C _ { l ^ { \prime } } \cap _ { l ^ { \prime } \neq l } C _ { l } = \emptyset$ 且  $D = \bigcup _ { l = 1 } ^ { k } C _ { l }$。相应地，我们用 $\lambda _ { j } \in \{ 1,2 , \ldots , k \}$ 表示样本 $x_j$ 的簇标记（cluster label），即样本 属于这个簇：$x_j \in C_{\lambda_j}$  ，于是聚类的结果可用包含 $m$ 个元素的簇标记向量 $\boldsymbol { \lambda } = \left( \lambda _ { 1 } ; \lambda _ { 2 } ; \ldots ; \lambda _ { m } \right)$  ：即样本 $x_j$是属于标记为 $\lambda_j$ 簇的。



### 1.2 性能度量



聚类的性能度量称为聚类“有效性指标”（validity index）。



物以类聚，我们想要结果“**簇内相似度**”高，“**簇间相似度低**”。



聚类性能度量分两类：

1. 结果与**参考模型**比较，称为“**外部指标**”
2. 直接考察结果，**不利用参考模型**，称为“**内部指标**”



#### 1.2.1 外部指标

数据集 $D = \left\{ \boldsymbol { x } _ { 1 } , \boldsymbol { x } _ { 2 } , \ldots , \boldsymbol { x } _ { m } \right\}$，簇划分 $\mathcal { C } = \{ C _ { 1 }, C _ { 2 } , \ldots , C _ { k } \}$，外部的**参考模型**给出的簇为：$\mathcal { C } ^ { * } = \left\{ C _ { 1 } ^ { * } , C _ { 2 } ^ { * } , \ldots , C _ { s } ^ { * } \right\}$。相应地，$\lambda$ 和 $\lambda^*$  表示 $\mathcal { C }$ 和 $\mathcal { C^* }$ 对应的簇标记向量。



将样本两两配对，定义：
$$
a = | S S | , \quad S S = \left\{ \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) | \lambda _ { i } = \lambda _ { j } , \lambda _ { i } ^ { * } = \lambda _ { j } ^ { * } , i < j \right) \}
$$

$$
b = | S D | , \quad S D = \left\{ \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) | \lambda _ { i } = \lambda _ { j } , \lambda _ { i } ^ { * } \neq \lambda _ { j } ^ { * } , i < j \right) \}
$$

$$
c = | D S | , \quad D S = \left\{ \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) | \lambda _ { i } \neq \lambda _ { j } , \lambda _ { i } ^ { * } = \lambda _ { j } ^ { * } , i < j \right) \}
$$

$$
d = | D D | , \quad D D = \left\{ \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) | \lambda _ { i } \neq \lambda _ { j } , \lambda _ { i } ^ { * } \neq \lambda _ { j } ^ { * } , i < j \right) \}
$$



$a, b, c, d$ 四个集合代表四种样本对，比如 $a$ 是在  $\mathcal { C }$ 和  $\mathcal { C^* }$ 中都是隶属相同簇的样本对集合。由于每个样本对$(x_i, x_j)  (i<j)$ 只能出现在一个集合中，因此有 $a + b + c + d = m ( m - 1 ) / 2$



我们有以下这些常用的聚类性能度量**外部指标**：



**$Jaccard$ 系数（简称 $JC$）**
$$
\mathrm { JC } = \frac { a } { a + b + c }
$$


**$FM$ 指数 （简称 $FMI$）**


$$
\mathrm { FMI } = \sqrt { \frac { a } { a + b } \cdot \frac { a } { a + c } }
$$


$Rand$ 指数 

$$
\mathrm { RI } = \frac { 2 ( a + d ) } { m ( m - 1 ) }
$$


显然，上述性能度量结果在 $[0,1]$区间，值越大越好。


#### 1.2.2 内部指标

聚类结果 $\mathcal { C } = \left\{ C _ { 1 } , C _ { 2 } , \ldots , C _ { k } \right\}$，定义：

$$
\operatorname { avg } ( C ) = \frac { 2 } { | C | ( | C | - 1 ) } \sum _ { 1 \leq i < j \leq | C | } \operatorname { dist } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right)
$$

$$
\operatorname { diam } ( C ) = \max _ { 1 \leqslant i < j \leqslant | C | } \operatorname { dist } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right)
$$

$$
d _ { \min } \left( C _ { i } , C _ { j } \right) = \min _ { x _ { i } \in C _ { i } , x _ { j } \in C _ { j } } \operatorname { dist } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right)
$$

$$
d _ { \operatorname { cen } } \left( C _ { i } , C _ { j } \right) = \operatorname { dist } \left( \boldsymbol { \mu } _ { i } , \boldsymbol { \mu } _ { j } \right)
$$


$\operatorname { dist } ( \cdot , \cdot )$ 用于计算两个样本之间的距离；

**$\mu$** 代表簇 $C$ 的中心点 $\mu =\frac { 1 } { | C | } \sum _ { 1 \leqslant i \leqslant | C | } \boldsymbol { x } _ { i }$。


**$\operatorname { avg } ( C )$** 对应于簇 $C$ 内样本间的平均距离

**$diam(C)$** 对应于簇 $C$ 内样本间的最远距离

**$d _ { \mathrm { cen } } \left( C _ { i } , C _ { j } \right)$** 对应于簇 $C_i$ 和 $C_j$ 中心点之间的距离


下面是常用的聚类性能度量内部指标：

**$DB$ 指数（$DBI$）**

$$
\mathrm { DBI } = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \max _ { j \neq i } \left( \frac { \operatorname { avg } \left( C _ { i } \right) + \operatorname { avg } \left( C _ { j } \right) } { d _ { \operatorname { cen } } \left( \boldsymbol { \mu } _ { i } , \boldsymbol { \mu } _ { j } \right) } \right)
$$


**$Dumm$ 指数（$DI$）**

$$
\mathrm { DI } = \min _ { 1 \leqslant i \leq k } \left\{ \min _ { j \neq i } \left( \frac { d _ { \min } \left( C _ { i } , C _ { j } \right) } { \max _ { 1 \leqslant l \leqslant k } \operatorname { diam } \left( C _ { l } \right) } \right) \right\}
$$

显然 $DBI$ 值越小越好，$DI$ 值越大越好。



### 1.3 距离计算

对函数 $\operatorname { dist } ( \cdot , \cdot )$，若其为一个“距离度量”，则要满足以下性质：

- 非负性：$\operatorname { dist } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) \geq 0$
- 同一性：$\operatorname { dist } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) = 0$ 当且仅当 $\boldsymbol { x } _ { i } = \boldsymbol { x } _ { j }$
- 对称性：$\operatorname { dist } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) = \operatorname { dist } \left( \boldsymbol { x } _ { j } , \boldsymbol { x } _ { i } \right)$
- 直递性：$\operatorname { dist } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) \leqslant \operatorname { dist } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { k } \right) + \operatorname { dist } \left( \boldsymbol { x } _ { k } , \boldsymbol { x } _ { j } \right)$

给定样本 $\boldsymbol { x } _ { i } = \left( x _ { i 1 } ; x _ { i 2 } ; \ldots ; x _ { i n } \right)$ 与 $\boldsymbol { x } _ { j } = \left( x _ { j 1 } ; x _ { j 2 } ; \ldots ; x _ { j n } \right)$

**闵可夫斯基距离（Minkowski distance）：**

$$
\operatorname { dist } _ { \mathrm { mk } } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) = \left( \sum _ { u = 1 } ^ { n } \left| x _ { i u } - x _ { j u } \right| ^ { p } \right) ^ { \frac { 1 } { p } }
$$

即 $L_p$ 范数，$\left\| \boldsymbol { x } _ { i } - \boldsymbol { x } _ { j } \right\| _ { \boldsymbol { p } }$

$p = 2$ 时，即欧氏距离（Euclidean distance）

$$
\operatorname { dist } _ { \mathrm { ed } } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) = \left\| \boldsymbol { x } _ { i } - \boldsymbol { x } _ { j } \right\| _ { 2 } = \sqrt { \sum _ { u = 1 } ^ { n } \left| x _ { i u } - x _ { j u } \right| ^ { 2 } }
$$

$p=1$ 时，即曼哈顿距离（Manhattan distance）

$$
\operatorname { dist } _ { \operatorname { man } } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) = \left\| \boldsymbol { x } _ { i } - \boldsymbol { x } _ { j } \right\| _ { 1 } = \sum _ { u = 1 } ^ { n } \left| x _ { i u } - x _ { j u } \right|
$$


属性划分有**连续属性**和**离散属性**。对于离散属性，在讨论距离计算时，属性上是否定义“序”关系至关重要，比如像 `{1,2,3}` 的离散属性与连续属性的性质更接近，能直接在属性上计算距离：“1”与“2”较近，“1”与“3”较远，这种属性称为“**有序属性**”，而像`{飞机,火车,轮船}`这样的理算属性不能直接计算九黎，称为“**无序属性**”。

显然，闵可夫斯基距离可用于有序属性。

对无序属性可采用 VDM (Value Difference Metric)，令 $m _ { u , a }$ 表示在属性 $u$ 上取值为 $a$ 的样本数，$m_{u,a,i}$ 表示第 $i$ 个样本簇中在属性 $u$ 上取值为 $a$ 的样本数，$k$ 为样本簇数，则属性 $u$ 上两个离散值 $a$ 和 $b$ 之间的 VDM 距离为：

$$
\operatorname { VDM } _ { p } ( a , b ) = \sum _ { i = 1 } ^ { k } \left| \frac { m _ { u , a , i } } { m _ { u , a } } - \frac { m _ { u , b , i } } { m _ { u , b } } \right| ^ { p }
$$

于是，将闵可夫斯基距离和 VDM 结合可处理混合属性，嘉定有 $n_c$ 个有序属性，$n - n_c$ 个无序属性，令有序属性排在无序属性前：

$$
\operatorname { MinkovDM } _ { p } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) = \left( \sum _ { u = 1 } ^ { n _ { c } } \left| x _ { i u } - x _ { j u } \right| ^ { p } + \sum _ { u = n _ { c } + 1 } ^ { n } \operatorname { VDM } _ { p } \left( x _ { i u } , x _ { j u } \right) \right) ^ { \frac { 1 } { p } }
$$


不同属性重要性不同时，可使用**加权距离**，以加权闵可夫斯基距离为例：

$$
\operatorname { dist } _ { \mathrm { wmk } } \left( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } \right) = \left( w _ { 1 } \cdot \left| x _ { i 1 } - x _ { j 1 } \right| ^ { p } + \ldots + w _ { n } \cdot \left| x _ { i n } - x _ { j n } \right| ^ { p } \right) ^ { \frac { 1 } { p } }
$$

权重 $w _ { i } \geqslant 0 ( i = 1,2 , \dots , n )$ 表征不同属性的重要性，通常和为 1


需要注意，现在的距离是来定义**相似度度量**，而不是“**距离度量**”，因为距离不一定满足所有距离度量的性质，特别是“**直递性**”，称为“**非度量距离**”。现实中有必要基于数据样本来确定合适的距离计算式，可通过“距离度量学习”来实现。

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fvv6jfw43zj30rq0fq42e.jpg)


## 2.  聚类算法



### 2.1 原型聚类

prototype-based clustering

原型聚类假设聚类结构能通过一组原型刻画，在现实聚类中极为常用。

通常算法先对原型进行初始化，然后对原型进行迭代跟新求解。


#### k-means 算法

样本集 $D = \left\{ \boldsymbol { x } _ { 1 } , \boldsymbol { x } _ { 2 } , \ldots , \boldsymbol { x } _ { m } \right\}$, 簇划分 $\mathcal { C } = \left\{ C _ { 1 } , C _ { 2 } , \ldots , C _ { k } \right\}$，k-均值算法最小化平方误差：

$$
E = \sum _ { i = 1 } ^ { k } \sum _ { \boldsymbol { x } \in C _ { i } } \left\| \boldsymbol { x } - \boldsymbol { \mu } _ { i } \right\| _ { 2 } ^ { 2 }
$$

其中 $\boldsymbol { \mu } _ { i } = \frac { 1 } { \left| C _ { i } \right| } \sum _ { \boldsymbol { x } \in C _ { i } } \boldsymbol { x }$ 是簇 $C_i$ 的均值向量。

**上式刻画了簇内样本围绕簇均值向量的紧密程度**，$E$ 值越小，则簇内样本相似度越高。

最小化上式不容易，找到其最优解要考察 D 所有可能的簇划分，是一个 NP 难问题。k 均值算法采用了贪心策略，通过迭代优化来近似求解 $E = \sum _ { i = 1 } ^ { k } \sum _ { \boldsymbol { x } \in C _ { i } } \left\| \boldsymbol { x } - \boldsymbol { \mu } _ { i } \right\| _ { 2 } ^ { 2 }$

算法如下：

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fvv7232qg2j312m0skqc5.jpg)


![](https://ws2.sinaimg.cn/large/006tNbRwgy1fvvacf347dj31280y2aoc.jpg)


**sklearn API**
```python
sklearn.cluster.KMeans(n_clusters=8,
	init='k-means++', 
	n_init=10, 
	max_iter=300, 
	tol=0.0001, 
	precompute_distances='auto', 
	verbose=0, 
	random_state=None, 
	copy_x=True, 
	n_jobs=1, 
	algorithm='auto'
)
# n_clusters:簇的个数，即你想聚成几类
# init: 初始簇中心的获取方法
# n_init: 获取初始簇中心的更迭次数，为了弥补初始质心的影响，算法默认会初10次质心，实现算法，然后返回最好的结果。
# max_iter: 最大迭代次数（因为kmeans算法的实现需要迭代）
# tol: 容忍度，即kmeans运行准则收敛的条件
# precompute_distances：是否需要提前计算距离，这个参数会在空间和时间之间做权衡，如果是True 会把整个距离矩阵都放到内存中，auto 会默认在数据样本大于featurs*samples 的数量大于12e6 的时候False,False 时核心实现的方法是利用Cpython 来实现的
# verbose: 冗长模式
# random_state: 随机生成簇中心的状态条件。
# copy_x: 对是否修改数据的一个标记，如果True，即复制了就不会修改数据。bool 在scikit-learn 很多接口中都会有这个参数的，就是是否对输入数据继续copy 操作，以便不修改用户的输入数据。这个要理解Python 的内存机制才会比较清楚。
# n_jobs: 并行设置
# algorithm: kmeans的实现算法，有：‘auto’, ‘full’, ‘elkan’, 其中 'full’表示用EM方式实现

import numpy as np
from sklearn.cluster import KMeans

data = np.random.rand(100, 3) # 样本大小为100, 特征数为 3

# 构造一个聚类数为3的聚类器
estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(data)
label_pred = estimator.labels_ # 标签
centroids = estimator.cluster_centers_ # 聚类中心
inertia = estimator.inertia_ # 每个点到中心的距离平方和
```



#### 学习向量量化 (LVQ) 算法

学习向量量化（Learning Vector Quantization）和 k-means 类似，也是试图找一组原型向量来刻画聚类结构，但是不同的是，LVQ 假设数据带有类别标记，**学习过程利用样本的监督信息**来辅助聚类。

样本集带有辅助标记：$D = \left\{ \left( \boldsymbol { x } _ { 1 } , y _ { 1 } \right) , \left( \boldsymbol { x } _ { 2 } , y _ { 2 } \right) , \ldots , \left( \boldsymbol { x } _ { m } , y _ { m } \right) \right\}$, LVQ 的目标是学得一组 n 维原型 $\left\{ \boldsymbol { p } _ { 1 } , \boldsymbol { p } _ { 2 } , \dots , \boldsymbol { p } _ { q } \right\}$，每个原型向量代表一个聚类簇，簇标记 $t_i \in \mathcal { Y }$

![](https://ws1.sinaimg.cn/large/006tNbRwgy1fvvai1w7nnj30vw0ok101.jpg)

解释一下：

第一行原型向量初始化，如对第 q 个簇可以从**类别标记**为 $t_q$ 的样本中随机选取一个作为原型向量。

每一次迭代，算法随机选择样本 $\left( \boldsymbol { x } _ { j } , y _ { j } \right)$ ，计算与它最近的原型向量 $p_{i*}$，再观察 $y_i$ 和$p_{i*}$的类别是否一致，如果一致，则令 $p_{i*}$ 向 $x_j$ 反方向靠拢

$$
\boldsymbol { p } ^ { \prime } = \boldsymbol { p } _ { i ^ { * } } + \eta \cdot \left( \boldsymbol { x } _ { j } - \boldsymbol { p } _ { i ^ { * } } \right)
$$

不一致，向相反方向走远：

$$
\boldsymbol { p } ^ { \prime } = \boldsymbol { p } _ { i ^ { * } } - \eta \cdot \left( \boldsymbol { x } _ { j } - \boldsymbol { p } _ { i ^ { * } } \right)
$$


在学习一组原型向量 $\left\{ \boldsymbol { p } _ { 1 } , \boldsymbol { p } _ { 2 } , \dots , \boldsymbol { p } _ { q } \right\}$ 后， 即可实现对样本空间 $\mathcal { X }$ 的粗划分，每个样本被划入与其距离最近的原型向量所代表的簇中

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fvvc14mmkbj31280ye4ew.jpg)



#### 高斯混合聚类

与 k均值，LVQ 用**原型向量**来刻画聚类结构不同，高斯混合（Mixture-of-Gaussian）聚类采用**概率模型**来表达聚类原型。

**多元高斯分布**：

$n$ 维向量 $x$ 若服从高斯分布，则其概率密度为

$$
p ( \boldsymbol { x } ) = \frac { 1 } { ( 2 \pi ) ^ { \frac { n } { 2 } } | \mathbf { \Sigma } | ^ { \frac { 1 } { 2 } } } e ^ { - \frac { 1 } { 2 } ( \boldsymbol { x } - \boldsymbol { \mu } ) ^ { \mathrm { T } } \mathbf { \Sigma } ^ { - 1 } ( \boldsymbol { x } - \boldsymbol { \mu } ) }
$$

$\mu$ 是 $n$ 维均值向量，$\Sigma$ 是 $n \times n$ 协方差矩阵

概率密度函数可定记为 $p ( \boldsymbol { x } | \boldsymbol { \mu } , \boldsymbol { \Sigma } )$，可定义**高斯混合分布**

$$
p _ { \mathcal { M } } ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { k } \alpha _ { i } \cdot p ( \boldsymbol { x } | \boldsymbol { \mu } _ { i } , \mathbf { \Sigma } _ { i } )
$$




该分布由 k 个混合分布组成，每个混合成分对应一个高斯分布，$\alpha_i > 0$ 为相应的混合系数，$\sum _ { i = 1 } ^ { k } \alpha _ { i } = 1$.

假设样本的生成过程由高斯混合分布给出：首先，根据 $\alpha _ { 1 } , \alpha _ { 2 } , \dots , \alpha _ { k }$ 定义的先验分布，选择高斯混合成分，其中 $\alpha_i$ 为选择第 $i$ 个混合成分的概率；然后根据被选择的混合成分的概率密度函数进行采样，从而生成相应的样本。

若训练集 $D = \left\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { m } \right\}$ 由上述过程生成，令随机变量 $z_j \in {1,2,...k}$ 表示生成样本 $x_j$ 的高斯混合成分，其取值未知，显然 $z_j$ 的先验概率 $P \left( z _ { j } = i \right)$ 对应于 $\alpha _ { i } ( i = 1,2 , \dots , k )$，根据贝叶斯定理， $z_j$ 的后验分布对应于

$$
\left.\begin{aligned} p _ { \mathcal { M } } \left( z _ { j } = i | \boldsymbol { x } _ { j } \right) & = \frac { P \left( z _ { j } = i \right) \cdot p _ { \mathcal { M } } \left( \boldsymbol { x } _ { j } | z _ { j } = i \right) } { p _ { \mathcal { M } } \left( \boldsymbol { x } _ { j } \right) } \\ & = \frac { \alpha _ { i } \cdot p \left( \boldsymbol { x } _ { j } | \boldsymbol { \mu } _ { i } , \boldsymbol { \Sigma } _ { i } \right) } { \sum _ { l = 1 } ^ { k } \alpha _ { l } \cdot p \left( \boldsymbol { x } _ { j } | \boldsymbol { \mu } _ { l } , \mathbf { \Sigma } _ { l } \right) } \end{aligned} \right.
$$

$p _ { \mathcal { M } } \left( z _ { j } = i | \boldsymbol { x } _ { j } \right)$ 给出了：样本 $x_j$ 由第 $i$ 个高斯混合成分生成的的后验概率，简记为 $\gamma _ { j i } ( i = 1,2 , \dots , k )$

当高斯混合分布 $p _ { \mathcal { M } } ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { k } \alpha _ { i } \cdot p ( \boldsymbol { x } | \boldsymbol { \mu } _ { i } , \mathbf { \Sigma } _ { i } )$ 已知时，划分为 k 个簇 $\mathcal { C } = \left\{ C _ { 1 } , C _ { 2 } , \ldots , C _ { k } \right\}$，每个样本 $x_j$ 的簇标记 $\lambda_j$ 如下确定：

$$
\lambda _ { j } = \underset { i \in \{ 1,2 , \ldots , k \} } { \arg \max } \gamma _ { j i }
$$

从原型聚类的角度看，高斯混合聚类采用概率模型（高斯分布）对原型进行刻画，簇划分则由原型对应的后验概率确定。

$p _ { \mathcal { M } } ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { k } \alpha _ { i } \cdot p ( \boldsymbol { x } | \boldsymbol { \mu } _ { i } , \mathbf { \Sigma } _ { i } )$ 已知时，划分为 k 个簇 $\mathcal { C } = \left\{ C _ { 1 } , C _ { 2 } , \ldots , C _ { k } \right\}$ 的模型参数 $\left\{ \left( \alpha _ { i } , \boldsymbol { \mu } _ { i } , \boldsymbol { \Sigma } _ { i } \right) | 1 \leqslant i \leqslant k \right\}$ 如何求解？对样本集 D，可采用极大似然估计，集最大化（对数）似然

$$
\left.\begin{aligned} L L ( D ) & = \ln \left( \prod _ { j = 1 } ^ { m } p _ { \mathcal { M } } \left( \boldsymbol { x } _ { j } \right) \right) \\ & = \sum _ { j = 1 } ^ { m } \ln \left( \sum _ { i = 1 } ^ { k } \alpha _ { i } \cdot p \left( \boldsymbol { x } _ { j } | \boldsymbol { \mu } _ { i } , \boldsymbol { \Sigma } _ { i } \right) \right) \end{aligned} \right.
$$

常采用 EM 算法进行迭代优化求解：

若参数 $\left\{ \left( \alpha _ { i } , \boldsymbol { \mu } _ { i } , \boldsymbol { \Sigma } _ { i } \right) | 1 \leqslant i \leqslant k \right\}$ 能使得上式最大化，则由  $\frac { \partial L L ( D ) } { \partial \boldsymbol { \mu } _ { i } } = 0$ 有

$$
\sum _ { j = 1 } ^ { m } \frac { \alpha _ { i } \cdot p \left( \boldsymbol { x } _ { j } | \boldsymbol { \mu } _ { i } , \mathbf { \Sigma } _ { i } \right) } { \sum _ { l = 1 } ^ { k } \alpha _ { l } \cdot p \left( \boldsymbol { x } _ { j } | \boldsymbol { \mu } _ { l } , \mathbf { \Sigma } _ { l } \right) } \left( \boldsymbol { x } _ { j } - \boldsymbol { \mu } _ { i } \right) = 0
$$

$$
\boldsymbol { \mu } _ { i } = \frac { \sum _ { j = 1 } ^ { m } \gamma _ { j i } \boldsymbol { x } _ { j } } { \sum _ { j = 1 } ^ { m } \gamma _ { j i } }
$$

即各混合成分的均值可通过样本加权平均来估计，样本权重是每个样本属于该成分的后验概率，类似由 $\frac { \partial L L ( D ) } { \partial \mathbf { \Sigma } _ { i } } = 0$ 得：

$$
\boldsymbol { \Sigma } _ { i } = \frac { \sum _ { j = 1 } ^ { m } \gamma _ { j i } \left( \boldsymbol { x } _ { j } - \boldsymbol { \mu } _ { i } \right) \left( \boldsymbol { x } _ { j } - \boldsymbol { \mu } _ { i } \right) ^ { \mathrm { T } } } { \sum _ { j = 1 } ^ { m } \gamma _ { j i } }
$$


对于混合系数 $\alpha_i$，除了要最大化 $L L ( D )$，还需满足 $\alpha _ { i } \geqslant 0 , \sum _ { i = 1 } ^ { k } \alpha _ { i } = 1$，考虑 $L L ( D )$ 的拉格朗日形式

$$
L L ( D ) + \lambda \left( \sum _ { i = 1 } ^ { k } \alpha _ { i } - 1 \right)
$$

其中 $\lambda$ 为拉格朗日乘子，对$\alpha_i$ 的导数为0，有

$$
\sum _ { j = 1 } ^ { m } \frac { p \left( \boldsymbol { x } _ { j } | \boldsymbol { \mu } _ { i } , \boldsymbol { \Sigma } _ { i } \right) } { \sum _ { l = 1 } ^ { k } \alpha _ { l } \cdot p \left( \boldsymbol { x } _ { j } | \boldsymbol { \mu } _ { l } , \mathbf { \Sigma } _ { l } \right) } + \lambda = 0
$$

两边同乘以 $\alpha_i$ ，对所有样本求和可知 $\lambda=-m$，有

$$
\alpha _ { i } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \gamma _ { j i }
$$

即每个高斯成分的混合系数由样本属于该成分的平均后验概率确定。

由上述推导可获得高斯混合模型的 EM 算法：在每步迭代中，先根据当前参数来计算每个样本属于每个高斯分布的后验概率$\gamma j i$(E步)，再根据$\boldsymbol { \mu } _ { i } = \frac { \sum _ { j = 1 } ^ { m } \gamma _ { j i } \boldsymbol { x } _ { j } } { \sum _ { j = 1 } ^ { m } \gamma _ { j i } }$, $\mathbf { \Sigma } _ { i } = \frac { \sum _ { j = 1 } ^ { m } \gamma _ { j i } \left( \boldsymbol { x } _ { j } - \boldsymbol { \mu } _ { i } \right) \left( \boldsymbol { x } _ { j } - \boldsymbol { \mu } _ { i } \right) ^ { \mathrm { T } } } { \sum _ { j = 1 } ^ { m } \gamma _ { j i } }$，$\alpha _ { i } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \gamma _ { j i }$ 更新模型参数 $\left\{ \left( \alpha _ { i } , \boldsymbol { \mu } _ { i } , \boldsymbol { \Sigma } _ { i } \right) | 1 \leqslant i \leqslant k \right\}$ (M步)


![](https://ws2.sinaimg.cn/large/006tNbRwgy1fvvcs7qku9j30zw0v6tjy.jpg)

### 2.2 密度聚类



#### DBSCAN 算法

![](https://ws1.sinaimg.cn/large/006tNbRwgy1fvvctmmlaqj30xm0yuajq.jpg)



### 2.3 层次聚类

层次聚类（hierarchical clustering）在不同层次对数据集进行划分，从而采用树形的聚类结构。

数据集划分可自底向上聚合、或自顶向下分拆

#### AGNES 算法

AGNES 采用自底向上聚合策略，先将每个样本看作一个初始聚类簇，然后每一步找出距离最近的两个聚类簇进行合并，不断重复，直到达到预设的聚类簇个数。关键是如何计算聚类簇间的距离。

最小距离：$d _ { \min } \left( C _ { i } , C _ { j } \right) = \min _ { x \in C _ { i } , z \in C _ { j } } \operatorname { dist } ( \boldsymbol { x } , \boldsymbol { z } )$

最大距离：$d _ { \max } \left( C _ { i } , C _ { j } \right) = \max _ { \boldsymbol { x } \in C _ { i } , \boldsymbol { z } \in C _ { j } } \operatorname { dist } ( \boldsymbol { x } , \boldsymbol { z } )$

平均距离：$d _ { \mathrm { avg } } \left( C _ { i } , C _ { j } \right) = \frac { 1 } { \left| C _ { i } \right| \left| C _ { j } \right| } \sum _ { \boldsymbol { x } \in C _ { i } } \sum _ { \boldsymbol { z } \in \mathbb { C } _ { j } } \operatorname { dist } ( \boldsymbol { x } , \boldsymbol { z } )$

最小距离由两个簇的最近样本决定，最大距离由两个簇的 最远样本决定，平均距离由两个簇的 所有样本共同决定。当聚类簇由$d _ { \min }$、$d _ { \max }$、$d_{avg}$计算时，AGNES 算法相应地被称为“单链接”，“全连接”，“均链接” 算法





![](https://ws3.sinaimg.cn/large/006tNbRwgy1fvvcu21m17j30s20yojzk.jpg)

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fvvev612h5j310q0s8dk9.jpg)

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fvvevros3oj311g10gapk.jpg)










