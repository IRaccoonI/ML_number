# %%
import gzip
f = gzip.open('./train/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 60000

import numpy as np
import matplotlib.pyplot as plt
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

f = gzip.open('./train/train-labels-idx1-ubyte.gz','r')
f.read(8)
buf = f.read(num_images + 1)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)



images = map(lambda im: np.asarray(im).squeeze().reshape(28*28) / 255, data)
train_data = list(zip(images, labels))

def print_mtx(mtx):
    print(np.around(mtx, decimals=2))


#%%

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
            res_layers.append(
              1 / (1 + pow(np.e, -np.dot(self.weights[i], pre_layer)))
            )
            pre_layer = res_layers[-1]

        return res_layers

    def train(self, train_data, k = 20):
        len_data = len(train_data)
        for l in range(k):
            n = 1 / 2

            for tr_data, tr_lbl in train_data:
                cur_res = self(tr_data)

                goal_val = cur_res[-1][tr_lbl]
                prefer_last_layer = np.array(
                    [val if val < goal_val else goal_val for val in cur_res[-1]]
                )
                prefer_last_layer[int(tr_lbl)] = 1
                d_last_layer = prefer_last_layer - cur_res[-1]

                finded_d_layers = []
                finded_d_layers.insert(0, d_last_layer)

                for i in range(len(self.sizes_layers) - 2, 0, -1):
                    cur_weight_transp = np.transpose(self.weights[i])
                    prefer_weight = np.matmul(cur_weight_transp, d_last_layer)
                    d_last_layer = prefer_weight[:-1] if self._weights_offset else prefer_weight

                    finded_d_layers.insert(0, prefer_weight)


                d_weights = [
                    np.zeros(self.weights[i].shape)
                    for i in range(len(self.weights))
                ]

                for i in range(len(self.weights)):
                    cur_weight = self.weights[i]
                    cur_d_layer = finded_d_layers[i]
                    if self._weights_offset and i != len(self.weights) - 1:
                        cur_d_layer = cur_d_layer[:-1]

                    cur_res_layer = n * cur_d_layer * cur_res[i + 1] * (1 - cur_res[i + 1])
                    cur_res_layer = cur_res_layer[np.newaxis].T
                    cur_input_row = np.append(cur_res[i], 1)
                    cur_append_mtx = cur_input_row * cur_res_layer
                    d_weights[i] += cur_append_mtx + cur_weight

                self.weights = [
                    d_weights[i] + self.weights[i]
                    for i in range(len(self.weights))
                ]





machine = ML((28*28, 16, 16, 10, ))

test_im = np.asarray(data[0]).squeeze()
test_data = test_im.reshape(28*28) / 255
# plt.imshow(test_im)

def calc():
    res_machine = machine(test_data)[-1]
    print_mtx(res_machine)
    top = -1
    for i, val in enumerate(sorted(res_machine, reverse=True)):
        if val == res_machine[5]:
            top = i + 1
            break
    print(res_machine[5], top)
    print()

calc()

machine.train(train_data, 1)

calc()
# plt.imshow(image)
