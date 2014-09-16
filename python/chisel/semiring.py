"""
@author waziz
"""
import math
    
class CountSemiring(object):

    @staticmethod
    def sum(x, y):
        return x + y

    @staticmethod
    def product(x, y):
        return x * y

    @staticmethod
    def division(x, y):
        return float(x)/y

    @staticmethod
    def as_real(x):
        return float(x)
    
class SumTimesSemiring(object):

    @staticmethod
    def sum(x, y):
        return math.log(math.exp(x) + math.exp(y))

    @staticmethod
    def product(x, y):
        return x + y

    @staticmethod
    def division(x, y):
        return x - y

    @staticmethod
    def as_real(x):
        return math.exp(x)

class MaxTimesSemiring(object):

    @staticmethod
    def sum(x, y):
        return max(x, y)

    @staticmethod
    def product(x, y):
        return x + y

    @staticmethod
    def division(x, y):
        return x - y

    @staticmethod
    def as_real(x):
        return math.exp(x)

class ProbabilitySemiring(object):

    @staticmethod
    def sum(x, y):
        return x + y

    @staticmethod
    def product(x, y):
        return x * y

    @staticmethod
    def division(x, y):
        return x / y

    @staticmethod
    def as_real(x):
        return x
