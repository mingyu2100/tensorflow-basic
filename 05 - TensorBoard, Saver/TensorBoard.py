# https://github.com/golbin/TensorFlow-Tutorials/blob/master/05%20-%20TensorBoard%2C%20Saver/01%20-%20Saver.py

import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

    tf.summary.histogram('Weights', W1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)

    # tf.summary 모듈의 scalar 함수는 값이 하나인 텐서를 수집할 때 사용
    tf.summary.scalar('cost', cost)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all() # 앞서 지정한 텐서들을 수집
writer = tf.summary.FileWriter('./logs', sess.graph) # 그래프와 텐서들의 값을 저장할 디렉터리 설정

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step), 'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data}) # merged로 모아둔 텐서의 값들을 계산하여 수집
    writer.add_summary(summary, global_step=sess.run(global_step)) # 해당 값들을 앞서 지정한 디렉터리에 저장

# 최적화가 끝난 뒤 학습된 변수들을 지정한 체크포인트 파일에 저장
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))