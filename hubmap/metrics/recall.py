BLOOD_VESSEL_CLASS_INDEX = 0


class Recall:
    """
    Code based on: https://github.com/nikhilroxtomar/TransResUNet
    """

    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Recall"):
        self._name = name
    
    def __call__(self, prediction, target):
        intersection = (target * prediction).sum((-2, -1))
        result = (intersection + 1e-15) / (target.sum((-2, -1)) + 1e-15)
        return result[:, BLOOD_VESSEL_CLASS_INDEX]


