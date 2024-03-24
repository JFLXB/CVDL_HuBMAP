BLOOD_VESSEL_CLASS_INDEX = 0


class Acc:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Acc"):
        self._name = name
        
    def __call__(self, prediction, target):
        prediction_bv_mask = prediction[:, BLOOD_VESSEL_CLASS_INDEX, :, :]
        target_bv_mask = target[:, BLOOD_VESSEL_CLASS_INDEX, :, :]
        
        correct = (prediction_bv_mask == target_bv_mask).sum((-2, -1))
        total = target.size(-2) * target.size(-1)
        accuracy = correct / total
        return accuracy
    
    
