#coding: utf-8
# 将MNIST数据集读取出来，并保存为图片文件，更直观的理解MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
# scipy的misc作用是读取以数据转化并保存为图像
import scipy.misc
import os

# 读取MNIST数据集。如果不存在会事先下载到"MNIST_data/"文件夹中。可以到"MNIST_data/"文件夹下查看下载后的文件
# 原始表示：1  独热表示：0，1，0，0，0，0，0，0，0，0
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 我们把原始图片保存在MNIST_data/raw/文件夹下
# 如果没有这个文件夹会自动创建
save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前20张图片
for i in range(20):
    # 请注意，mnist.train.images[i, :]就表示第i张图片（序号从0开始）
    image_array = mnist.train.images[i, :]
    # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    # 这种图片序号的保存方法很不错
    filename = save_dir + 'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存。
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

print('Please check: %s ' % save_dir)

