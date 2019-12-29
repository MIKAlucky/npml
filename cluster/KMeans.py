import numpy as np

from base import ClusterMixin, UnsupervisedModel
from utils.distances import euclidean_distance


class KMeans(UnsupervisedModel, ClusterMixin):

    def fit(self, X: np.ndarray, k: int = 8, init_centriods_method="kmeans++", max_iter=100000,
            tol=1e-10, distance=euclidean_distance):
        """
        Parameters
        ----------
        X: 训练集, (n_samples, n_features)
        k: 将X聚成k类
        init_centriods_method: 初始化聚类中心的方式
            - random 随机选择样本X中的k个点作为初始聚类中心
            - kmeans++
                - 从输入的数据点集合中随机选择一个点作为第一个聚类中心
                - 对于数据集中的每一个点xi，计算它与已选择的聚类中心中最近聚类中心的距离d，
                  然后选择使得d最大的那个点xi作为下一个聚类中心
                - 重复以上两步骤，直到选择了k个聚类中心
        max_iter: 最大迭代次数，到达max_iter则停止迭代
        tol: 质心前后两次变化的最大误差小于tol则停止迭代
        distance: 距离函数
            - 欧氏距离
            - 曼哈顿距离
            - 切比雪夫距离
            - 闵可夫斯基距离
            - 标准化欧氏距离
            - 马氏距离
            - 巴氏距离
        Return
        ------
            self
                self.centers: 质心, (k)
                self.results: 存放每个点对应到哪个类, key为类，value为点的集合
        """
        # 初始化质心
        centriods = self._init_centriods(X, k, method=init_centriods_method)

    @staticmethod
    def _init_centriods(self, X: np.ndarray, k: int, method: str = "Kmeans++"):
        """初始化聚类中心"""
        if method not in ["Random", "KMeans++"]:
            raise ValueError("only supported method [Random, KMeans++]")
        elif method == "Random":  # 从X中随机选取k个点
            indices = np.random.choice(len(X), k, replace=False)
            self.centriods = X[indices]
        else:  # KMeans++ TODO
            centriods_indices = np.zeros(k)  # 初始化聚类中心点索引
            # 1、随机选择一个点作为第一个聚类中心
            first_index = np.random.choice(len(X), 1)[0]

            self.centriods = X[centriods_indices]
        return self.centriods

    @staticmethod
    def _get_nearest_class(sample, centers):
        """点sample离centers中哪个质心更近，返回哪个质心的索引 + 1"""
        return np.argmin(np.sqrt(np.sum((centers - sample) ** 2, axis=1))) + 1

    def predict(self, X):
        """预测
        Parameters
        ----------
        X: 需要预测的数据集，(n_samples, n_features)

        Returns
        ------
        数据集每个点分到的类, (n_samples)
        """
        return np.array([KMeans._get_nearest_class(x, self.centers) for x in X])
