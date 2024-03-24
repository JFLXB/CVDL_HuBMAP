from hubmap.metrics.precision import Precision
from hubmap.metrics.recall import Recall

BLOOD_VESSEL_CLASS_INDEX = 0


class F2:
    """
    Code based on: https://github.com/nikhilroxtomar/TransResUNet
    """

    @property
    def name(self):
        return self._name
    
    def __init__(self, name="F2", beta=2):
        self._name = name
        self._beta = beta
    
    def __call__(self, prediction, target):
        p = Precision()(prediction, target)
        r = Recall()(prediction, target)
        return (1+self._beta**2.) *(p*r) / float(self._beta**2*p + r + 1e-15)




