import json
import numpy as np



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

    def copy(self):
        n_machine = ML(self.sizes_layers)
        n_machine.weights = self.weights.copy()
        return n_machine

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
        total_iter = len_data * k
        sep_progress = int(total_iter / 100)
        for l in range(k):
            n = (len_data - l) / len_data * 4

            d_weights = [
                np.zeros(self.weights[i].shape)
                for i in range(len(self.weights))
            ]

            for ll, [tr_data, tr_lbl] in enumerate(train_data):
                if (l * len_data + ll) % sep_progress == 0:
                    print(str(int((l * len_data + ll) / total_iter * 100)) + '%')
                    pass

                cur_res = self(tr_data)

                goal_val = cur_res[-1][tr_lbl]
                prefer_last_layer = np.array(
                    [val
                    if val < goal_val - .3 else
                    val / 4 * 3
                    for val in cur_res[-1]]
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


                for i in range(len(self.weights)):
                    cur_d_layer = finded_d_layers[i]
                    if self._weights_offset and i != len(self.weights) - 1:
                        cur_d_layer = cur_d_layer[:-1]

                    cur_res_layer = n * cur_d_layer * cur_res[i + 1] * (1 - cur_res[i + 1])
                    cur_res_layer = cur_res_layer[np.newaxis].T
                    cur_input_row = np.append(cur_res[i], 1)
                    cur_append_mtx = cur_input_row * cur_res_layer
                    d_weights[i] += cur_append_mtx

            self.weights = [
                d_weights[i] / len_data + self.weights[i]
                for i in range(len(self.weights))
            ]

    def save_weight(self, f_name):
        with open(f_name, 'w') as out_f:
            json.dump([weight.tolist() for weight in self.weights], out_f)

    def load_weight(self, f_name):
        with open(f_name) as out_f:
            data = json.load(out_f)
        self.weights = [np.array(weight) for weight in data]
