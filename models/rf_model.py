import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ..interface import DigitClassificationInterface


class RFModel(DigitClassificationInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        # fit the model with random data
        dummy_X = np.random.rand(100, 724)
        dummy_y = np.random.randint(0, 10, 100)
        self.model.fit(dummy_X, dummy_y)

    def predict(self, image: np.ndarray) -> int:
        return self.model.predict(image)[0]

