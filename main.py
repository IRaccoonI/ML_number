# %%
import gzip
f = gzip.open('./train/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 60000

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)


import matplotlib.pyplot as plt
image = np.asarray(data[-1]).squeeze()
# plt.imshow(image)

f = gzip.open('./train/train-labels-idx1-ubyte.gz','r')
f.read(8)
buf = f.read(num_images + 1)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

class ML:
    def __init__(self, sizes_layers, *_, _random = True, _weights_offset = True):
        self._weights_offset = _weights_offset
        self.sizes_layers = sizes_layers
        self.weights = []
        _l_size = sizes_layers[0]
        for i, l_size in enumerate(sizes_layers[1:]):
            # random array of weights from -1 to 1
            weight_add = int(_weights_offset)
            if _random:
                n_weight_arr = np.random.rand(
                    l_size, _l_size + weight_add
                    ) * 2 - 1

            self.weights.append(n_weight_arr)

            _l_size = l_size
        # print(self.weights)

    def __call__(self, inputs):
        res_layers = [inputs]
        pre_layer = inputs
        for i in range(len(self.sizes_layers) - 1):
            if self._weights_offset:
                pre_layer = np.insert(pre_layer, -1, 1)
            print(self.weights[i].shape)
            print(pre_layer.shape)
            res_layers.append(
              1 / (1 + pow(np.e, -np.dot(self.weights[i], pre_layer)))
            )
            pre_layer = res_layers[-1]

        return res_layers

    def train():
        pass

inputs = np.reshape(image, 28*28)
ML((28*28, 16, 16, 10))(inputs / 255)