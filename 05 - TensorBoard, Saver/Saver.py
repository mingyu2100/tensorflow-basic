# https://github.com/golbin/TensorFlow-Tutorials/blob/master/05%20-%20TensorBoard%2C%20Saver/01%20-%20Saver.py

import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

global_step = tf.Variable(0, trainable=False, name='global_step')
"""
global_step : 학습에 직접 사용되지는 않고, 학습 횟수를 카운트하는 변수
이를 위해 변수 정의 시 trainable=False라는 옵션 추가.
"""

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
"""
tf.global_variables() : 앞서 정의한 변수들을 가져오는 함수
이 함수를 써서 앞서 정의한 변수들을 모두 가져와서, 이후 이 변수들을 파일에 저장하거나
이전에 학습한 결과를 불러와 담는 변수들로 사용한다.
"""

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
"""
./model 디렉터리에 기존에 학습해둔 모델이 있는지를 확인해서(
모델이 있다면 saver.restore 함수를 사용해 학습된 값들을 불러오고,
아니면 변수를 새로 초기화한다.(단순 확인만, 없다고 ./model을 생성하지 않는다.)
학습된 모델을 저장한 파일을 체크포인트 파일(ckeckpoint file)이라고 한다.
"""

for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step), 'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 최적화가 끝난 뒤 학습된 변수들을 지정한 체크포인트 파일에 저장
saver.save(sess, './model/dnn.ckpt', global_step=global_step)
"""
두 번째 매개변수는 체크포인트 파일의 위치와 이름
상위 디렉터리(./model)는 미리 생성돼 있어야 한다.(없으면 생성함)
global_step의 값은 저장할 파일의 이름에 추가로 붙게 되며,
텐서 변수 또는 숫자값을 넣어줄 수 있다.
"""

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))