import numpy as np
from numpy.linalg import pinv

from base import RegressorMixin


class OrdinaryLeastSquare(RegressorMixin):

    def fit(self, X, Y):
        """训练
        y = w0+w1x1+w2x2+...+wnxn
        W = (X^T X)^(-1)X^T Y

        Parameters
        ----------
        X：训练数据，二维数组
        Y：X的真实值，一维数组

        Returns
        -------

        """
        n_samples, n_features = X.shape

        # 给X添加一列1， 将y转换成(n_samples, 1) 便于计算
        X = np.concatenate((np.ones(n_samples).reshape((n_samples, 1)), X), axis=1)
        Y = Y.reshape((n_samples, 1))

        self.theta = pinv(X.T @ X) @ X.T @ Y  # A@B 等于 np.dot(A, B)
        self.intercept = self.theta[0, 0]  # 截距项
        self.coef = self.theta[1:, 0]  # 系数
        return self

    def predict(self, X):
        """预测
        Parameters
        ----------
        X: 待预测的二维数组

        Returns
        ------
        预测标签的一维数组
        """
        return X @ self.coef + self.intercept
