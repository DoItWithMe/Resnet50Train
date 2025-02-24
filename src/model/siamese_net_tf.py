import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import torch.nn as nn


class SiameseNetworkL2Net3UseTf(nn.Module):
    def __init__(self):
        super(SiameseNetworkL2Net3UseTf, self).__init__()
        self.model_url = (
            "https://www.kaggle.com/api/v1/models/google/resnet-v2/tensorFlow2/50-feature-vector/2/download"
        )

        self.resnet = tf.keras.Sequential(  # type: ignore
            [
                hub.KerasLayer(
                    self.model_url, input_shape=(224, 224, 3), trainable=True
                ),
            ]
        )
