import numpy as np
import torch
from models import CNNModel, RFModel, RandomModel


class DigitClassifier:
    def __init__(self, algorithm: str):
        self.algorithm = algorithm

        if algorithm == "cnn":
            self.model = CNNModel()
        elif algorithm == "rf":
            self.model = RFModel()
        elif algorithm == "rand":
            self.model = RandomModel()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. The options are: cnn, rf, rand")

    def predict(self, image: np.ndarray) -> int:
        if image.shape != (28,28):
            raise ValueError("Input image size must be 28x28")

        if self.algorithm == "cnn":
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            return self.model.predict(image)
        elif self.algorithm == "rf":
            image = image.flatten().reshape(1, -1)
            return self.model.predict(image)
        elif self.algorithm == "rand":
            image = image[9:19, 9:19] # center crop
            return self.model.predict(image)
