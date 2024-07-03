import random
import numpy as np
from models.interface import DigitClassificationInterface


class RandomModel(DigitClassificationInterface):
    def predict(self, image: np.ndarray) -> int:
        return random.randint(0,9)

