"""主成分分析"""

import numpy as np


class PCA():
    def __init__(self):
        super(PCA, self).__init__()
        self.model_type = "PCA"

    def fit(self, X: np.ndarray, dim: int = 2) -> object:
        """训练

        Parameters
        ----------
        X : 二维数组 (n_samples, n_features)
        dim : int, optional
            欲将数据缩放到几维
        """

        X = X - X.mean(axis=0)  # 均值归0
        cov = np.cov(X[:, 0], X[:, 1])  # 协方差矩阵
        eigenvalues, eigenvectors = np.linalg.eig(cov)  # 特征值，特征向量(每一列为1个特征向量)

        index = np.argsort(eigenvalues[::-1])
        eigenvectors = eigenvectors[:, index]  # 重新排列特征向量 从左往右按特征值大小降序
        # 再取排序后的前n_dim个特征向量, 每一列为一个特征
        self.eigenvectors = eigenvectors[:, :dim]
        return self

    def transform(self, X):
        return np.matmul(X, self.eigenvectors)
