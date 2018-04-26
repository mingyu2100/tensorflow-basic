# K-최근접 이웃(KNN) - 분류 회귀가 모두 가능한 지도 학습 알고리즘
# d = sqrt(square(x-y))

import tensorflow as tf
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/', one_hot=True)

train_pixels, train_list_values = mnist.train.next_batch(100)
test_pixels, test_list_of_values = mnist.test.next_batch(10)

train_pixel_tensor = tf.placeholder(tf.float32, [None, 784])
test_pixel_tensor = tf.placeholder(tf.float32, [784])

# 비용 함수와 최적화
# 비용 함수는 픽셀 간의 거리에 해당한다.
# reduction_indices=1은 axis=1과 동일. 요즘에는 사용하지 않는다.
distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor, tf.negative(test_pixel_tensor))), reduction_indices=1)

# 거리 함수를 최소화하기 위해 argmin 을 사용하며, 가장 작은 거리를 갖는 인덱스(최근접 이웃)를 리턴한다.
pred = tf.argmin(distance, 0)

accuracy = 0
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_list_of_values)):
        # pred 함수를 사용해 최근접 이웃 인덱스를 평가한다.
        nn_index = sess.run(pred, feed_dict={train_pixel_tensor: train_pixels, test_pixel_tensor: test_pixels[i, :]})

        # 최근접 이웃의 클래스 레이블을 확인하고, 실제 레이블과 비교한다.
        print('Test N ', i, 'Predicted Class: ', np.argmax(train_list_values[nn_index]), 'True Class: ', np.argmax(test_list_of_values[i]))
        if np.argmax(train_list_values[nn_index]) == np.argmax(test_list_of_values[i]):
            accuracy += 1./len(test_pixels)
        print('Result= ', accuracy)

