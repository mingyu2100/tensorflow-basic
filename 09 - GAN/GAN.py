# GAN 기본 모델 구현하기

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28 * 28
n_noise = 128

# 구분자에 넣을 이미지(실제 이미지, 생성한 가짜 이미지)
X = tf.placeholder(tf.float32, [None, n_input]) # 실제 이미지
Z = tf.placeholder(tf.float32, [None, n_noise]) # 가짜 이미지(노이즈에서 생성함으로)

# 생성자(Generator)에 사용할 변수들을 설정
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 구분자(Discriminator)에 사용할 변수들을 설정
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))
""""
구분자는 진짜와 얼마나 가까운가를 판단하는 값으로, 0~1 사이의 값을 출력하도록 함(하나의 스칼라값을 출력)
실제 이미지를 판별하는 구분자 신경망과 생성한 이미지를 판별하는 구분자 신경망은 같은 변수를 사용해야 한다.
같은 신경망으로 구분을 시켜야 진짜 이미지와 가짜 이미지를 구분하는 특징들을 동시에 잡아낼 수 있기 때문이다.
"""

# 생성자 신경망
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)

    return output

# 구분자 신경망
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2) # 0~1 사이의 스칼라값 하나를 출력

    return output

# 무작위한 노이즈 만들기
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

G = generator(Z)  # 노이즈(Z)를 이용해 가짜 이미지를 만들 생성자 G 생성
D_gene = discriminator(G)
D_real = discriminator(X)

# 생성자가 만든 이미지를 구분자가 '가짜'라고 판단하도록 하는 손실값(경찰 학습용)
# 진짜 이미지 판별값(D_real)->1, 가짜 이미지 판별값(D_gene)->0
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

# 생성자가 만든 이미지를 구분자가 '진짜'라고 판단하도록 하는 손실값(위조지폐범 학습용)
# 가짜 이미지 판별값(D_gene)->1
loss_G = tf.reduce_mean(tf.log(D_gene))
"""
GAN 학습은 loss_D, loss_G 모두를 최대화하는 것
단, loss_D, loss_G는 서로 연관되어 있어 두 손실값이 항상 같이 증가하는 경향을 보이지는 않음
loss_D 증가 -> loss_G 하락, loss_G 증가 -> loss_D 하락하는 경쟁 관계
"""

D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]
"""
loss_D를 구할 때는 구분자 신경망에 사용되는 변수들만 사용
loss_G를 구할 때는 생성자 신경망에 사용되는 변수들만 사용
그래야 loss_D를 학습할 때는 생성자가 변하지 않고, loss_G를 학습할 때는 구분자가 변하지 않음
"""

# loss를 최대화해야 하지만, 최적화에 쓸 수 있는 함수는 minimize뿐이므로 음수 부호를 붙임
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))

    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('optimization end.')
