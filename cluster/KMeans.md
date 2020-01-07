代价函数： 

$$J(c,u)=\sum_{i=1}^k||x(i)-u_{c^{(i)}}||^2$$

 $u_{c^{(i)}}$表示第i个类的均值。 我们希望代价函数最小，直观的来说，各类内的样本越相似，其与该类均值间的误差平方越小，对所有类所得到的误差平方求和，即可验证分为k类时，各聚类是否是最优的

# 算法流程

输入：待聚类的样本X、预先给定的聚类数K 输出：样本X中每个样本被分到的类、最终的所有聚类中心 流程：

1. 初始化K个聚类中心作为最初的中心
2. 循环每个样本，计算其与K个聚类中心的距离，将该样本分到距离最小的那个聚类中心
3. 将每个聚类中的样本均值作为新的聚类中心
4. 重复步骤2和3直到聚类中心不再变化

# 性能分析

- 优点

  - 是解决聚类问题的一种经典算法，简单、快速
  - 对处理大数据集，该算法是相对可伸缩和高效率的。它的复杂度是$O(nkt)$,其中, $n$是所有对象的数目, $k$ 是簇的数目, $t$ 是迭代的次数。通常$k<<n$ 且$t<<n$
  - 当结果簇是密集的，而簇与簇之间区别明显时, 它的效果较好

- 缺点

  - 在簇的平均值被定义的情况下才能使用，这对于处理符号属性的数据不适用
  - 必须事先给出K
  - 对初值敏感，对于不同的初始值，可能会导致不同结果
  - ÷如何初始化聚类中心

- 随机选择K个样本点

- KMeans++

  1. 从输入的数据点集合中随机选择一个点作为第一个聚类中心𝜇1
  2. 对于数据集中的每一个点𝑥𝑖，计算它与已选择的聚类中心中最近聚类中心的距离$𝐷(𝑥𝑖)=𝑎𝑟𝑔𝑚𝑖𝑛||𝑥_𝑖−𝜇_𝑟||_2^2$, 𝑟=1,2,...𝑘𝑠𝑒𝑙𝑒𝑐𝑡𝑒𝑑  <!--_-->
  3. 选择一个新的数据点作为新的聚类中心，选择的原则是：𝐷(𝑥)较大的点，被选取作为聚类中心的概率较大
  4. 重复2和3直到选择出k个聚类质心
  5. 利用这k个质心来作为初始化质心去运行标准的K-Means算法

# 如何确认聚类数K

## 肘方法确定最佳聚类数K

K_means参数的最优解是以成本函数最小化为目标

成本函数为各个类畸变程度之和

每个类的畸变程度等于该类重心与其内部成员位置距离的平方和

但是平均畸变程度会随着K的增大先减小后增大，所以可以求出最小的平均畸变程度

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# 1 数据可视化
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T
plt.figure()
plt.axis([0, 5, 0, 5])
plt.grid(True)
plt.plot(X[:, 0], X[:, 1], 'k.')
plt.show()
 
# 2 肘部法求最佳K值
K = range(1, 10)
mean_distortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    mean_distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, metric='euclidean'), axis=1)) / X.shape[0])
plt.plot(K, mean_distortions, 'bx-')
plt.xlabel('k')
# 以下为解决中文乱码方法
font = FontProperties(fname=r'c:\windows\fonts\msyh.ttc', size=20)
plt.ylabel(u'平均畸变程度', fontproperties=font)
plt.title(u'用肘部法确定最佳的K值', fontproperties=font)
plt.show()
```

![](images/肘方法.jpeg)