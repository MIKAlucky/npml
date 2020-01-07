class ParametersError(Exception):
    """参数错误"""


class NotFittedError(Exception):
    """模型未训练"""


if __name__ == '__main__':
    a = ParametersError("123")
    raise a
