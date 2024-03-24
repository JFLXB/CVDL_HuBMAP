BLOOD_VESSEL_CLASS_INDEX = 0


class DiceScore:
    """
    Code based on: https://github.com/nikhilroxtomar/TransResUNet
    """

    @property
    def name(self):
        return self._name
    
    def __init__(self, name="DiceScore"):
        self._name = name
    
    def __call__(self, prediction, target):
        result = (2 * (target * prediction).sum((-2, -1)) + 1e-15) / (target.sum((-2, -1)) + prediction.sum((-2, -1)) + 1e-15)
        return result[:, BLOOD_VESSEL_CLASS_INDEX]


