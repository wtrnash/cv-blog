# Week 1 - Decision Tree

## 0.Entropy
>**Def** measurement of random variable's uncertainty

>**Equ** empirical entropy H(X), empirical conditional entropy H(Y|X)
>&#8195;&#8195;**$H(Y|X) = H(X,Y) - H(X)$**
>$= -\sum_{x,y}^{}P(X,Y) logP(X,Y)+\sum_{x}^{}P(X)logP(X)$
>$=-\sum_{x,y}^{}P(X,Y) logP(X,Y)+\sum_{x}^{}(\sum_{y}^{}P(X,Y))logP(X)$
>$=-\sum_{x,y}^{}P(X,Y) logP(X,Y+\sum_{x,y}^{}P(X,Y)logP(X)$
>$=-\sum_{x,y}^{}log\frac{P(X,Y)}{P(X)}$
>$=-\sum_{x,y}^{}logP(Y|X)$
>$=-\sum_{x}^{}\sum_{y}^{}P(X)P(Y|X)logP(Y|X)$
>>$=-\sum_{x}^{}P(X)\sum_{y}^{}P(Y|X)logP(Y|X)$
>
>$=\sum_{x}^{}P(X)H(Y|X=x_{i})$

>**Ann** binary logarithm



## 1. Feature Selection
### &#8195;1.1 information gain
>**Def** mutual information of data set and feature

>**Equ** feature A in $(a_{1},a_{2},...,a_{n})$, data set D, information gain g, empirical entropy H, class number K
>&#8195;&#8195;**$g(D,A) = H(D) - H(D|A)$**
>$=-\sum_{k=1}^{K}P(C_{k})logP(C_{k})+\sum_{i=1}^{n}P(A_{i})\sum_{k=1}^{K}P(D_{k}|A_{i})logP(D_{k}|A_{i})$
>$=-\sum_{k=1}^{K}\frac{\left |C_{k}\right |}{\left|D\right|}log\frac{\left |C_{k}\right |}{\left|D\right|}+\sum_{i=1}^{n}\frac{\left |D_{i}\right |}{\left|D\right|}\sum_{k=1}^{K}\frac{\left |D_{ik}\right |}{\left|D_{i}\right|}log\frac{\left |D_{ik}\right |}{\left|D_{i}\right|}$

### &#8195;*1.2 information gain ratio*
>**Def** Normalization: tackling the possibility of a overwhelming variable set

>**Equ** $g_{R}(D,A)=\frac{g(D,A)}{H_{A}(D)}$
>$=\frac{g(D,A)}{-\sum_{i=1}^{n}\frac{\left |D_{i}\right |}{\left|D\right|}log\frac{\left |D_{i}\right |}{\left|D\right|}}$

### &#8195;*1.3 Gini coefficient*
>**Def** purity of the data set

>**Equ** $Gini(D)=\sum_{i\neq j}^{}P{i}P{j}$
>$\because binaryTree$
>$=\sum_{k=1}^{K}P_{k}(1-P_{k}))$
>$=1-\sum_{k=1}^{K}(P_{k})^{2}$
>$=1-\sum_{k=1}^{K}(\frac{\left |C_{k}\right |}{\left|D\right|})^{2}$

>**Equ** $Gini(D,A)=\frac{\left |D_{1}\right |}{\left|D\right|}Gini(D1)+\frac{\left |D_{2}\right |}{\left|D\right|}Gini(D2)$

## 2.Generation
>**Def** 
> &#8195; &#8195;  $ID3$ ----  information gain
> &#8195; &#8195;  $C4.5$ ----  information gain ratio
>$1)$ calculate each gain ,  choose the maximum gain. 
>$2)$ divide recursively until (subtree in same class) or (gain $\leq$ threshold)

> &#8195; &#8195;  $CART$ ---- Gini coefficient
> $1)$ choose the min $Gini(D,A_{i})$ as the optimal segmentation point
> $2)$ divide recursively until (each feature traversed) or (subtree in same class) 

## 3.Pruning
>**Def** alleviate degree of overfitting
>**loss functon** a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. 

>**Equ** leaf node number $\left |T \right|$, class number $N_{t}$, parameter $\alpha$
>$C_{\alpha}(T)=\sum_{t=1}^{\left |T  \right |}N_{t}H_{t}(T)+\alpha\left |T \right|$
>$=-\sum_{t=1}^{\left |T  \right |}\sum_{k=1}^{K}N_{tk}log\frac{N_{tk}}{N_{t}}+\alpha\left |T \right|$
>decreasing loss ---- need pruning
>
![在这里插入图片描述](https://img-blog.csdn.net/20181001190025572?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzOTA3NDA4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

>**$CART$ $Pruning$**
>Situation: pruning without a given parameter $\alpha$
>In the way of loss function, leafy trees benefit from little $\alpha$ and the other way around.
>Pruning happens when loss function of a inner node acting as a root  equals to the one acting as a leaf.
>$C_{\alpha}(t)=C(t)+\alpha$
>$C_{\alpha}(T_{t})=C(T_{t})+\alpha\left |T \right|$
>$\because C_{\alpha}(t)=C_{\alpha}(T_{t})$
>$\therefore \alpha=\frac{C(t)-C(T_{t})}{\left |T \right|-1}$
> $1)$ each inner node has an $\alpha$
> $2)$ acquire an ascending set of $\alpha$ , and a set of subtrees accordingly
> $3)$ Cross Validation: test the subtree set, select the one with the highest accuracy rate
