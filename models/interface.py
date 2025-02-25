from abc import ABC, abstractmethod
import numpy as np

class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image) -> int:
        pass