"""PPQ Core Decorator & MetaClass definitions PPQ 核心装饰器、元类型定义.

You are not allowed to modify this 请勿修改此文件
"""

import gc
from typing import Callable

from torch.cuda import empty_cache


class SingletonMeta(type):
    """The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.

    see also: https://refactoring.guru/design-patterns/singleton/python/example
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Possible changes to the value of the `__init__` argument do not
        affect the returned instance."""
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def empty_ppq_cache(func: Callable):
    """Using empty_ppq_cache decorator to clear ppq memory cache, both gpu
    memory and cpu memory will be clear via this function.

    Function which get decorated by this will clear all ppq system cache BEFORE
    its running.

    Args:
        func (Callable): decorated function
    """

    def _wrapper(*args, **kwargs):
        empty_cache()
        gc.collect()
        return func(*args, **kwargs)

    return _wrapper


def ppq_quant_param_computing_function(func: Callable):
    """mark a function to be a scale-computing function.

    Args:
        func (Callable): decorated function
    """

    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper
