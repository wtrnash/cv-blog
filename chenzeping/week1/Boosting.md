# Boosting

> version: 1.0
>
> - 理论待完善
>
> - 代码待跟进

[TOC]

## 理论基础

### 名词解释

集成学习：通过构建并结合多个学习器来完成学习任务。

同质 homogeneous：决策树集成中全是决策树，神经网络集成中全是神经网络。

基学习器 base learner：同质集成中的个体学习器。

基学习算法 base learning algorithm：基学习器所使用的学习算法。

异质 heterogenous：集成包含不同类型的个体学习器。

组件学习器 component learner：和基学习器对应，它们统称为个体学习器。

### 集成学习的可行性证明

假设二分类问题 $y \in \{ - 1 , + 1 \}$ 和真实函数 $f$ ，假定基分类器的错误率是 $\epsilon$ ，即对每个**基分类器 $h_{i}$ **有：
$$
P \left( h _ { i } ( \boldsymbol { x } ) \neq f ( \boldsymbol { x } ) \right) = \epsilon
$$
假设集成通过投票结合 $T$ 个基分类器，若有超过半数的基分类器正确，则集成分类就正确：
$$
H ( \boldsymbol { x } ) = \operatorname { sign } \left( \sum _ { i = 1 } ^ { T } h _ { i } ( \boldsymbol { x } ) \right)
$$
根据 Hoeffding 不等式，得到集成后的错误率：
$$
\left.\begin{aligned} P ( H ( \boldsymbol { x } ) \neq f ( \boldsymbol { x } ) ) & = \sum _ { k = 0 } ^ { \lfloor T / 2 \rfloor } \left( \begin{array} { l } { T } \\ { k } \end{array} \right) ( 1 - \epsilon ) ^ { k } \epsilon ^ { T - k } \\ & \leqslant \exp \left( - \frac { 1 } { 2 } T ( 1 - 2 \epsilon ) ^ { 2 } \right) \end{aligned} \right.
$$

> $P ( H ( n ) \leqslant k )$ 是另一种写法，含义相同。
>
> 由这条表达式，我们有：
> $$
> h _ { i } ( x ) = \left\{ \begin{array} { c c } { 1 } & { C _ { n } ^ { x } p ^ { x } ( 1 - p ) ^ { n - x } > = 0.5 } \\ { - 1 } & { C _ { n } ^ { x } p ^ { x } ( 1 - p ) ^ { n - x } < 0.5 } \end{array} \right.
> $$
>

第一个等号表示 $n$ 个基学习器中分类正确的个数小于 $k$ 的概率。若假定集成通过简单投票法结合 $n$ 个分类器，超过半数的基学习器正确，则集成分类就正确，即 $k=0.5*n=(1−ϵ−\delta)n$ 。

第二个等号的 Hoeffding 不等式的定义，$δ > 0$ ：
$$
P ( H ( n ) \leqslant ( p - \delta ) n ) \leqslant e ^ { - 2 \delta ^ { 2 } n }
$$
其中 $\left( \begin{array} { l } { T } \\ { k } \end{array} \right)$ 表示 $C_{T} ^{k}$ ，$\delta = 0.5-\epsilon$ 。

当 $\epsilon >=0.5$ 时，上式不成立。随着集成中个体分类器数目 $T$ 的增大，集成的错误率将指数级下降，最终趋向于零。

> 在现实中，个体学习器是解决同一个问题训练出来的，它们不可能相互独立，如何生成不同的个体学习器，是集成学习研究的核心。

根据个体学习器的生成方式，目前集成学习方法大致分为两大类：**个体学习期之间存在强依赖关系、必须串行生成的序列化方法—— Boosting**；**个体学习器之间不能存在强依赖关系、可同时生成的并行化方法—— Bagging**。

## Boosting

> 先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本进行调整，使得先前基学习器出错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器。

### AdaBoost

只适用二分类任务，比较容易理解的是基于“线性模型” additive model ，即基学习器的线性组合：
$$
H ( \boldsymbol { x } ) = \sum _ { t = 1 } ^ { T } \alpha _ { t } h _ { t } ( \boldsymbol { x } )
$$
最小化指数损失函数：
$$
\ell _ { \mathrm { exp } } ( H | \mathcal { D } ) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H ( \boldsymbol { x } ) } \right]
$$
$f(x)$ 只有两个结果，1 或 -1 ：
$$
\ell _ { \mathrm { exp } } ( H | \mathcal { D } ) = e^{-H(x)}P(f(x)=1)+e^{H(x)}P(f(x)=-1)
$$
若 $H(x)$ 可以将损失函数最小化，那么对它求偏导：
$$
\frac { \partial \ell _ { \operatorname { exp } } ( H | \mathcal { D } ) } { \partial H ( \boldsymbol { x } ) } = - e ^ { - H ( \boldsymbol { x } ) } P ( f ( \boldsymbol { x } ) = 1 | \boldsymbol { x } ) + e ^ { H ( \boldsymbol { x } ) } P ( f ( \boldsymbol { x } ) = - 1 | \boldsymbol { x } )
$$
令上式为0，可以解出：
$$
H ( \boldsymbol { x } ) = \frac { 1 } { 2 } \ln \frac { P ( f ( x ) = 1 | \boldsymbol { x } ) } { P ( f ( x ) = - 1 | \boldsymbol { x } ) }
$$
因此：
$$
\begin{aligned}\operatorname { sign } ( H ( \boldsymbol { x } ) ) &= \operatorname { sign } \left( \frac { 1 } { 2 } \ln \frac { P ( f ( x ) = 1 | \boldsymbol { x } ) } { P ( f ( x ) = - 1 | \boldsymbol { x } ) } \right)\\&=\left\{ \begin{array} { l l } { 1 , } & { P ( f ( x ) = 1 | \boldsymbol { x } ) > P ( f ( x ) = - 1 | \boldsymbol { x } ) } \\ { - 1 , } & { P ( f ( x ) = 1 | \boldsymbol { x } ) < P ( f ( x ) = - 1 | \boldsymbol { x } ) } \end{array} \right.\\&=\underset { y \in \{ - 1,1 \} } { \arg \max } P ( f ( x ) = y | \boldsymbol { x } )\end{aligned}
$$
我们发现，指数损失函数最小化，则分类错误率也将最小，即达到了贝叶斯最优错误率。

当基分类器得到以后，该基分类器的权重 $a_{t}$ 应该使得 $a_{t}h_{t}$ 最小化指数损失函数:
$$
\ell _ { \exp } \left( \alpha _ { t } h _ { t } | \mathcal { D } _ { t } \right)= e ^ { - \alpha _ { t } } \left( 1 - \epsilon _ { t } \right) + e ^ { \alpha _ { t } } \epsilon _ { t }
$$
其中 $\epsilon _ { t } = P _ { x \sim \mathcal { D } _ { t } } \left( h _ { t } ( \boldsymbol { x } ) \neq f ( \boldsymbol { x } ) \right)$ ，考虑指数损失函数的导数：
$$
\frac { \partial \ell _ { \exp } \left( \alpha _ { t } h _ { t } | \mathcal { D } _ { t } \right) } { \partial \alpha _ { t } } = - e ^ { - \alpha _ { t } } \left( 1 - \epsilon _ { t } \right) + e ^ { \alpha _ { t } } \epsilon _ { t }
$$
上式为0，可以得到**权重更新公式**：
$$
\alpha _ { t } = \frac { 1 } { 2 } \ln \left( \frac { 1 - \epsilon _ { t } } { \epsilon _ { t } } \right)
$$

------

AdaBoost 算法在下一轮基学习中纠正错误，那么：
$$
\ell _ { \exp } \left( H _ { t - 1 } + h _ { t } | \mathcal { D } \right) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) \left( H _ { t - 1 } ( \boldsymbol { x } ) + h _ { t } ( \boldsymbol { x } ) \right) } \right]
$$
！！！它可以进行泰勒展开，同时注意到 $f ^ { 2 } ( x ) = h _ { t } ^ { 2 } ( x ) = 1$ ：
$$
\begin{aligned}\ell _ { \exp } \left( H _ { t - 1 } + h _ { t } | \mathcal { D } \right) &\simeq \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } \left( 1 - f ( \boldsymbol { x } ) h _ { t } ( \boldsymbol { x } ) + \frac { f ^ { 2 } ( \boldsymbol { x } ) h _ { t } ^ { 2 } ( \boldsymbol { x } ) } { 2 } \right) \right]\\&=\mathbb { E } _ { x \sim \mathcal { D } } \left[ e ^ { - f ( x ) H _ { t - 1 } ( x ) } \left( 1 - f ( x ) h _ { t } ( x ) + \frac { 1 } { 2 } \right) \right]\end{aligned}
$$
理想的基学习器：
$$
\begin{aligned}h _ { t } ( \boldsymbol { x } ) &= \underset { \boldsymbol { h } } { \arg \min } \ell _ { \exp } \left( H _ { t - 1 } + h | \mathcal { D } \right)\\&=\underset { h } { \arg \min } \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } \left( 1 - f ( \boldsymbol { x } ) h ( \boldsymbol { x } ) + \frac { 1 } { 2 } \right) \right]\\&=\underset { h } { \arg \max } \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } f ( \boldsymbol { x } ) h ( \boldsymbol { x } ) \right]\\&=\underset { h } { \arg \max } \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } [ \frac { e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } } { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } [ e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } )  }] } f ( \boldsymbol { x } ) h ( \boldsymbol { x } ) ]\end{aligned}
$$
因为 $\mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } \right]$ 是一个常数，令 $\mathcal { D } _ { t }$ 表示一个分布：
$$
\mathcal { D } _ { t } ( \boldsymbol { x } ) = \frac { \mathcal { D } ( \boldsymbol { x } ) e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } } { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { w } ) } \right] }
$$
这等价于令：
$$
\left.\begin{aligned} h _ { t } ( \boldsymbol { x } ) & = \underset { \boldsymbol { h } } { \arg \max } \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ \frac { e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } } { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } \right] } f ( \boldsymbol { x } ) h ( \boldsymbol { x } ) \right] \\ & = \underset { \boldsymbol { h } } { \arg \max } \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { t } } [ f ( \boldsymbol { x } ) h ( \boldsymbol { x } ) ] \end{aligned} \right.
$$
由于 $f ( x ) , h ( x ) \in \{ - 1 , + 1 \}$ ，有：
$$
f ( \boldsymbol { x } ) h ( \boldsymbol { x } ) = 1 - 2 \mathbb { I } ( f ( \boldsymbol { x } ) \neq h ( \boldsymbol { x } ) )
$$
那么理想的基学习器：
$$
h _ { t } ( \boldsymbol { x } ) = \underset { h } { \arg \min } \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { t } } [ \mathbb { I } ( f ( \boldsymbol { x } ) \neq h ( \boldsymbol { x } ) ) ]
$$
**样本分布更新公式**：
$$
\begin{aligned}\mathcal { D } _ { t + 1 } ( \boldsymbol { x } ) &= \frac { \mathcal { D } ( \boldsymbol { x } ) e ^ { - f ( \boldsymbol { x } ) H _ { t } ( \boldsymbol { x } ) } } { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t } ( \boldsymbol { x } ) } \right] }\\&= \frac { \mathcal { D } ( \boldsymbol { x } ) e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } e ^ { - f ( \boldsymbol { x } ) \alpha _ { t } h _ { t } ( \boldsymbol { x } ) } } { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t } ( \boldsymbol { x } ) } \right] }\\&=\mathcal { D } _ { t } ( \boldsymbol { x } ) \cdot e ^ { - f ( \boldsymbol { x } ) \alpha _ { t } h _ { t } ( \boldsymbol { x } ) } \frac { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t - 1 } ( \boldsymbol { x } ) } \right] } { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \left[ e ^ { - f ( \boldsymbol { x } ) H _ { t } ( \boldsymbol { x } ) } \right] }\end{aligned}
$$

> 重赋权法：根据样本分布为每个样本重新赋予一个权重。
>
> 重采样法：根据样本分布对训练集重新采样，再用采样样本集对基学习器进行训练。
>
> Boosting 主要关注降低偏差，因此 Boosting 能基于泛化性能相当弱的学习器构建很强的集成。

### 提升树

以决策树为基函数的提升方法被称为提升树，对分类问题决策树是二叉分类树，对回归问题决策树是二叉回归树。提升树可以表示为决策树的加法模型：
$$
f _ { M } ( x ) = \sum _ { m = 1 } ^ { M } T \left( x ; \Theta _ { m } \right)
$$
其中 $T \left( x ; \Theta _ { n } \right)$ 表示决策树；$\Theta _ { n }$ 表示决策树的参数； $M$ 是树的个数。

提升树算法采用**前向分步算法**。首先确定初始提升树 $f_{0}(x)=0$ ，第 m 步的模型是：
$$
f _ { m } ( x ) = f _ { m - 1 } ( x ) + T \left( x ; \Theta _ { m } \right)
$$
通过**经验风险极小化**确定下一棵决策树的参数：
$$
\hat { \Theta } _ { m } = \arg \min _ { \boldsymbol { \theta } _ { \boldsymbol { e } } } \sum _ { i = 1 } ^ { N } L \left( y _ { i } , f _ { m - 1 } \left( x _ { i } \right) + T \left( x _ { i } ; \Theta _ { m } \right) \right)
$$

> 这里的 $T$ 指的就是下一棵决策树。

不同问题的提升树学习算法，主要区别在于使用的损失函数不同，平方误差损失函数的回归问题，指数损失函数的分类问题。下面叙述回归问题的提升树：
$$
T ( x ; \Theta ) = \sum _ { j = 1 } ^ { J } c _ { j } I \left( x \in R _ { j } \right)
$$
x 是输入， y 是输出，c 是输出常量，J 是回归树的复杂度即叶节点的个数，$\Theta = \left\{ \left( R _ { 1 } , c _ { 1 } \right) , \left( R _ { 2 } , c _ { 2 } \right) , \cdots , \left( R _ { J } , c _ { J } \right) \right\}$ 表示树的区域划分和各区域上的常数。

回归问题提升树使用以下前向分布算法：
$$
f _ { 0 } ( x ) = 0
$$

$$
f _ { m } ( x ) = f _ { m - 1 } ( x ) + T \left( x ; \Theta _ { m } \right) , \quad m = 1,2 , \cdots , M
$$

$$
f _ { M } ( x ) = \sum _ { m = 1 } ^ { M } T \left( x ; \Theta _ { m } \right)
$$

在前向分布算法的第 m 步，给定当前模型 $f_{m-1}(x)$ ，需求解：
$$
\hat { \Theta } _ { m } = \arg \min _ { \Theta _ { m } } \sum _ { i = 1 } ^ { N } L \left( y _ { i } , f _ { m - 1 } \left( x _ { i } \right) + T \left( x _ { i } ; \Theta _ { m } \right) \right)
$$
当使用平方误差损失函数时：
$$
L ( y , f ( x ) ) = ( y - f ( x ) ) ^ { 2 }
$$

$$
\left.\begin{aligned} L \left( y , f _ { m - 1 } ( x ) + T \left( x ; \Theta _ { m } \right) \right) & = \left[ y - f _ { m - 1 } ( x ) - T \left( x ; \Theta _ { m } \right) \right] ^ { 2 } \\ & = \left[ r - T \left( x ; \Theta _ { m } \right) \right] ^ { 2 } \end{aligned} \right.
$$

这里， $r = y - f _ { m - 1 } ( x )$ ，是当前模型拟合数据的残差。

**回归问题的提升树算法**：

输入：训练数据集 $T={(x1,y1),(x2,y2),...(xn,yn)}, xi, yi$ 

输出：提升树 $f_{M}(x)$ 

1. 初始化 $f_{0}(x)=0$ 
2. 开始循环 m = 1,2,...M
3. 计算残差：$r _ { m i } = y _ { i } - f _ { m - 1 } \left( x _ { i } \right) , \quad i = 1,2 , \cdots , N$ 
4. 拟合残差 $r_{mi}$ 学习一个回归树，得到 $T \left( x ; \Theta _ { m } \right)$ 
5. 更新 $f _ { m } ( x ) = f _ { m - 1 } ( x ) + T \left( x ; \Theta _ { m } \right)$ 
6. 得到回归问题提升树 $f _ { M } ( x ) = \sum _ { m = 1 } ^ { M } T \left( x ; \Theta _ { m } \right)$ 

### 梯度提升

这是利用最速下降法的近似方法，其关键是利用损失函数的负梯度在当前模型的值：
$$
- \left[ \frac { \partial L \left( y , f \left( x _ { i } \right) \right) } { \partial f \left( x _ { i } \right) } \right] _ { f ( x ) = f _ { m + 1 } ( x ) }
$$
作为回归问题提升树算法中的残差的近似值。

输入：训练数据集 $T={(x1,y1),(x2,y2),...(xn,yn)}, xi, yi$ ；损失函数 $L(y,f(x))$ 

输出：回归树 $\hat { f } ( x )$ 

1. 初始化：$f _ { 0 } ( x ) = \arg \min _ { c } \sum _ { i = 1 } ^ { N } L \left( y _ { i } , c \right)$ 
2. 开始循环 m 从 1 到 M
3. 对于 i 从 1 到 N ，计算： $r _ { m l } = - \left[ \frac { \partial L \left( y _ { i } , f \left( x _ { i } \right) \right) } { \partial f \left( x _ { i } \right) } \right] _ { f ( x ) = f _ { m- 1 } ( x ) }$ 
4. 对 $r_{mi}$ 拟合一个回归树，得到第 m 棵树的叶节点区域 $R_{mj}$
5. 对 j 从 1 到 J ，计算：$c _ { m j } = \arg \min _ { c } \sum _ { x _ { i } \in R _ { mj } } L \left( y _ { i } , f _ { m - 1 } \left( x _ { i } \right) + c \right)$ 
6. 更新 $f _ { m } ( x ) = f _ { m - 1 } ( x ) + \sum _ { j = 1 } ^ { J } c _ { m j } I \left( x \in R _ { m j } \right)$ 
7. 得到回归树 $\hat { f } ( x ) = f _ { M } ( x ) = \sum _ { m = 1 } ^ { M } \sum _ { j = 1 } ^ { J } c _ { m j } I \left( x \in R _ { m j } \right)$ 

### XGBoost

原始的 GBDT 算法基于经验损失函数的负梯度来构造新的决策树，只是在决策树构建完成后再进行剪枝，而 XGBoost 在决策树构建阶段就加入了正则化：
$$
L_{t} = \sum _ { i = 1 } ^ { n } l \left( y _ { i } , F_{t-1}(x_{i})+f_{t}(x_{i}) \right) + \sum _ { k = 1 } ^ { K } \Omega \left( f _ { k } \right)
$$
正则化定义：
$$
\Omega \left( f _ { t } \right) = \gamma T + \frac { 1 } { 2 } \lambda \sum _ { j = 1 } ^ { T } w _ { j } ^ { 2 }
$$
其中 T 是决策树 $f_{t}$ 中叶子节点的个数， $w_{j}$ 是第 j 个叶子节点的预测值，该损失函数在 $F_{t-1}$ 处进行二阶泰勒展开：
$$
L _ { i } \approx \overline { L } _ { r } = \sum _ { j = 1 } ^ { T } [ G _ { j } w _ { j } + \frac { 1 } { 2 } ( H _ { j } + \lambda ) w _ { j } ^ { 2 } ] + \gamma T
$$
这里 G 是一阶导，H 是二阶导，通过将损失函数对 $w_{j}$ 的导数为 0 ，可以求出在最小化损失函数的情况下各个叶子节点上的预测值：
$$
w _ { j } ^ { * } = - \frac { G _ { j } } { H _ { j } + \lambda }
$$
将预测值带入损失函数可以得到损失函数的最小值：
$$
\tilde { L } _ { t } ^ { * } = - \frac { 1 } { 2 } \sum _ { j = 1 } ^ { T } \frac { G _ { j } ^ { 2 } } { H _ { j } + \lambda } + \gamma T
$$
分裂前后损失函数的差值：
$$
Gain= \frac { G _ { L } ^ { 2 } } { H _ { L } + \lambda } + \frac { G _ { R } ^ { 2 } } { H _ { R } + \lambda } - \frac { ( G _ { L } + G _ { R } ) ^ { 2 } } { H _ { L } + H _ { R } + \lambda } - \gamma
$$
XGBoost 就是最大化这个差来进行决策树的构建，总的来说， XGBoost 和 GDBT 的差别和联系：

- GDBT 是机器学习算法， XGBoost 是该算法的工程实现。

- XGBoost 加入了正则化，支持多种类型的基分类器，支持对数据采样（和 RF 类似），能对缺省值处理。

## 算法实现

### AdaBoost 伪码

```python
"""
训练集 D = {(x1, y1), (x2, y2)..., (xm, ym)}
基学习算法 L
训练轮数 T
"""
D[1] = 1/m # 初始化样本权值分布
for t in range(T):
  h[t] = L(D, D[t]) # Dt 是数据分布
  e[t] = P(ht(x), f(x)) # 分类器 ht 的误差， ht(x) 是预测结果， f(x) 是真实结果
  if e[t] > 0.5:
    break
  a[t] = 0.5*np.log((1-e[t])/e[t])
  D[t+1] = D[t] / Z[t] # Zt 是规范化因子，以确保 D[t+1] 是一个分布
  if (h[t](x) == f(x)) D[t+1] *= exp(-a[t])
  else D[t+1] *= exp(a[t])
```

 最终返回 $H(x) = \operatorname { sign } \left( \sum _ { t = 1 } ^ { T } \alpha _ { t } h _ { t } ( \boldsymbol { x } ) \right)$ 。

> sign 表达符号函数。

## 经典题目

XGBoost 与 GBDT 的联系与区别有哪些？

从方差和偏差的角度解释 Boosting 和 Bagging  的原理？