# 학습시킨 모델을 저장하고 재사용하기
# 예측 값과 실제 값이 틀린 경우, 그림 출력하기

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)
# print(mnist.test.images.shape, mnist.test.labels.shape) # (10000, 784), (10000, 10)

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32, [None, 784]) # 28x28 픽셀
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32) # 학습 시에는 0.8, 예측 시에는 1

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.random_normal([256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
b2 = tf.Variable(tf.random_normal([256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
b3 = tf.Variable(tf.random_normal([10], stddev=0.01))
model = tf.matmul(L2, W3) + b3

# model = tf.nn.softmax(model)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 앞에서 구성한 신경망 모델을 초기화하고 학습을 진행할 세션을 시작
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
"""
미니배치(minibatch) : 이미지를 하나씩 학습시키는 것보다 여러 개를 한꺼번에 학습시키는 쪽이 효과가 좋다.
단, 많은 메모리와 높은 컴퓨팅 성능이 필요.
따라서 일반적으로 데이터를 적덩한 크기로 잘라서 학습시키는 것
"""

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)  # 미니배치가 총 몇 개인지를 저장

# 에포크(epoch) : 학습 데이터 전체를 한 바퀴 도는 것
# for epoch in range(30): # dropout 사용시 30. 미사용시 15. -> dropout 사용하면 학습이 느리게 진행되기 때문.
#     total_cost = 0
#
#     for i in range(total_batch):
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#
#         # 학습 코드: keep_prob를 0.8로 넣어준다.
#         _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
#         total_cost += cost_val
#
#     print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
# print('optimization end.')
# saver.save(sess, './model/MNIST', global_step=global_step)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 예측 코드: keep_prob를 1로 넣어준다.
print('accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

# print(W3)
# print(model)
# print(sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})[0])
# print(is_correct)
# print(sess.run(is_correct, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

labels = sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})

fig = plt.figure()
temp = 0
for i in range(mnist.test.num_examples):
    # 2행 5열의 그래프를 만들고, i + 1번째에 숫자 이미지를 출력
    subplot = fig.add_subplot(2, 5, temp + 1)
    # 이미지를 깨끗하게 출력하기 위해 x, y의 눈금 출력 X
    subplot.set_xticks([])
    subplot.set_yticks([])
    if np.argmax(labels[i]) != np.argmax(mnist.test.labels[i]):
        subplot.set_title('%d' % np.argmax(labels[i]))
        subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)
        temp += 1
        if temp >= 10:
            break

plt.show()
