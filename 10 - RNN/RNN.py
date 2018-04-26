# MNIST 를 RNN 으로

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

learning_rate = 0.001
total_epoch = 30
batch_size = 128

"""
RNN 은 순서가 있는 자료를 다루므로,
한 번에 입력받는 갯수와, 총 몇 단계로 이루어져있는 데이터를 받을지 설정해야한다.
가로 픽셀수를 n_input 으로, 세로 픽셀수를 입력 단계인 n_step 으로 설정한다.
"""
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# RNN 에 학습에 사용할 셀 생성
# BasicRNNCell, BasicLSTMCell, GRUCell
# n_hidden 개의 출력값을 갖는 RNN 셀 생성
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

# RNN 신경망 생성
# 기존 방식
# states = tf.zeros(batch_size)
# for i in range(n_step):
#     outputs, states = cell(X[:, i], states)
# ...
# 함수 이용
# tf.nn.dynamic_rnn(RNN 셀, 입력값, 입력값의 자료형)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# outputs : [batch_size, n_step, n_hidden]
#        -> [n_step, batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])  # (dynamic_rnn 의 time_major=True 로 해도 같은 결과)
#        -> [batch_size, n_hidden]
outputs = outputs[-1]

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # X data 를 RNN 입력 데이터에 맞게 [batch_size, n_step, n_input] 형태로 변환.
        batch_xs = batch_xs.reshape(batch_size, n_step, n_input)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('Optimization end.')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('Accuracy:', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))
