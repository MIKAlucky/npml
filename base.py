from metrics.classification import accuracy, precision, recall, f1, roc_auc, hinge
from metrics.regression import r2, mae, mse, explained_variance, max_error, msle, median_absolute_error
from types import FunctionType
from typing import Callable
import numpy as np


class Model(object):
    """模型类 本module下所有模型的基类"""

    def predict(self, *args, **kwargs):
        """"""

    def score(self, *args, **kwargs):
        """"""


class SupervisedModel(Model):
    """监督模型"""

    def fit(self, *args, **kwargs):
        """监督和半监督模型才需要训练 无监督不需要"""


class UnsupervisedModel(Model):
    """无监督模型"""


class SemiSupervisedModel(Model):
    """半监督模型"""

    def fit(self, *args, **kwargs):
        """"""


class BaseMixin:
    """所有Mixin类的基类"""
    _model_type = 'Model'

    def score(self, X: np.ndarray, y: np.ndarray, scorer: Callable[[np.ndarray, np.ndarray], float]) -> float:
        """模型在X,y上的评分

        @param X: 样本数据
        @param y: 样本值
        @param scorer: 评分函数 从metrics
        @return: 评分
        """
        y_predict = self.predict(X)
        score = scorer(y, y_predict)
        return score


class ClassifierMixin(BaseMixin):
    """所有分类模型的Mixin类"""
    _model_type = 'Classifier'


class RegressorMixin(BaseMixin):
    """所有回归模型的Mixin类"""
    _model_type = 'Regressor'


class ClusterMixin(BaseMixin):
    """所有聚类模型的Mixin类"""
    _model_type = 'Clusterer'
