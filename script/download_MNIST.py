__author__ = "Keenan Manpearl"
__date__ = "2023/1/24"

"""
original code by Xinqiang Ding <xqding@umich.edu>
download MNIST data from Yann Lecun's website 
[LeCun et al., 1998a]
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. 
        "Gradient-based learning applied to document recognition." 
        Proceedings of the IEEE, 86(11):2278-2324, November 1998.
"""

import numpy as np
import urllib3
import gzip
import pickle

# download train labels
print("Downloading train-labels-idx1-ubyte ......")
http = urllib3.PoolManager()
r = http.request('GET', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
data = gzip.decompress(r.data)
# number of images is 4-8
num = int.from_bytes(data[4:8], 'big')
# images start at 8
offset = 8 
# get labels from data 
train_label = np.array([data[offset+i] for i in range(num)])

# download train image
print("Downloading train-image-idx3-ubyte ......")
http.clear()
r = http.request('GET', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
data = gzip.decompress(r.data)
# number of images is 4:8 (60000)
num = int.from_bytes(data[4:8], 'big')
# number of rows is 8:12 (28)
nrows = int.from_bytes(data[8:12], 'big')
# number of cols is 12:16 (28)
ncols = int.from_bytes(data[12:16], 'big')
image = np.zeros((num, nrows * ncols))
# images start at 16
offset = 16
for k in range(num):
    for i in range(nrows):
        for j in range(ncols):
            image[k, i*ncols+j] = data[16 + k*nrows*ncols + i*ncols+j]
            # 0 = white 255 = black 
            # will give numbers between 0 and 1
train_image = image / 255.0
            
## download test labels
print("Downloading t10k-labels-idx1-ubyte ......")
http.clear()
r = http.request('GET', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
data = gzip.decompress(r.data)
num = int.from_bytes(data[4:8], 'big')
offset = 8
test_label = np.array([data[offset+i] for i in range(num)])

## download test image
print("Downloading t10k-image-idx3-ubyte ......")
http.clear()
r = http.request('GET', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
data = gzip.decompress(r.data)
num = int.from_bytes(data[4:8], 'big')
nrows = int.from_bytes(data[8:12], 'big')
ncols = int.from_bytes(data[12:16], 'big')
test_image = np.zeros((num, nrows * ncols))
offset = 16
for k in range(num):
    for i in range(nrows):
        for j in range(ncols):
            test_image[k, i*ncols+j] = data[16 + k*nrows*ncols + i*ncols+j]
            
test_image = test_image / 255.0

print("Saving data into a pickle file ...")
data = {'train_image': train_image,
        'train_label': train_label,
        'test_image': test_image,
        'test_label': test_label,}
with open("./data/MNIST.pkl", 'wb') as file_handle:
    pickle.dump(data, file_handle)
