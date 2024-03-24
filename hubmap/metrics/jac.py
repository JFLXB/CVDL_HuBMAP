BLOOD_VESSEL_CLASS_INDEX = 0


class Jac:
    """
    Code based on: https://github.com/nikhilroxtomar/TransResUNet
    """
    
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Jac"):
        self._name = name
    
    def __call__(self, prediction, target):
        intersection = (target * prediction).sum((-2, -1))
        union = target.sum((-2, -1)) + prediction.sum((-2, -1)) - intersection
        jac = (intersection + 1e-15) / (union + 1e-15)
        return jac[:, BLOOD_VESSEL_CLASS_INDEX]
