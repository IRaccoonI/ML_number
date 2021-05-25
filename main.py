# %%
import gzip

import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
import json


def load_data(im_path, lb_path, num_images):
    f = gzip.open(im_path,'r')
    image_size = 28
    f.read(16)

    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    f = gzip.open(lb_path,'r')
    f.read(8)
    buf = f.read(num_images + 1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)



    images = map(lambda im: np.asarray(im).squeeze().reshape(28*28) / 255, data)
    return list(zip(images, labels))

train_data = load_data(
    './train/train-images-idx3-ubyte.gz',
    './train/train-labels-idx1-ubyte.gz',
    60000
)

test_data = load_data(
    './train/t10k-images-idx3-ubyte.gz',
    './train/t10k-labels-idx1-ubyte.gz',
    10000
)

def print_mtx(mtx):
    print(np.around(mtx, decimals=2))

#%%
for i in range(20):
    plt.imshow(test_data[i][0].reshape((28, 28,)))


#%%
from ml import ML


b_machine = ML((28*28, 16, 16, 10, ))
b_machine.load_weight('weights.out')
# b_machine.save_weight('weights.out')

#%%

train_len = 5000
test_len = 10000 if train_len > 10000 else train_len
machine = b_machine.copy()
machine.load_weight('weights_m.out')
machine.train(train_data[:train_len], 80)
# machine.save_weight('weights_m.out')


print()

avg_pos = 0
b_avg_pos = 0

for i in range(test_len):
    cur_data, cur_lb = test_data[i]
    b_res = b_machine(cur_data)[-1]
    res = machine(cur_data)[-1]
    top_pos = sorted(res, reverse=True).index(res[cur_lb]) + 1
    b_top_pos = sorted(b_res, reverse=True).index(b_res[cur_lb]) + 1

    avg_pos += top_pos
    b_avg_pos += b_top_pos

print(b_avg_pos / test_len)
print(avg_pos / test_len)
print()

#%%
machine.save_weight('weights_m.out')