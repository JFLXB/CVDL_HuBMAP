BLOOD_VESSEL_CLASS_INDEX = 0


class Confidence:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Confidence"):
        self._name = name
        
    def __call__(self, probs):
        # Get the maximum probability over all classes
        max_probs, _ = probs.max(dim=1)
        # Select the blood vessel probabilities
        blood_vessel_probs = max_probs[:, BLOOD_VESSEL_CLASS_INDEX]
        # Calculate the mean probability over all pixels
        mean_prob = blood_vessel_probs.mean()
        return mean_prob


