# 단어 자동 완성

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# one-hot 인코딩 사용 및 디코딩을 하기 위해 연관 배열을 만든다.
# {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k': 10, ...}
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# wor -> X, d -> Y
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']


def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        # 입력값용으로, 단어의 처음 세 글자의 알파벳 인덱스를 구한 배열을 만든다.[22 14 17]
        input = [num_dic[n] for n in seq[:-1]]

        # 출력값용으로, 마지막 글자의 알파벳 인덱스를 구한다.(ex. deep -> 'p': 15)
        target = num_dic[seq[-1]]

        # 입력값을 원-핫 인코딩으로 변환한다.
        input_batch.append(np.eye(dic_len)[input])

        target_batch.append(target)

    return input_batch, target_batch


learning_rate = 0.01
n_hidden = 128
total_epoch = 30

# RNN 을 구성하는 시퀀스의 갯수(단어의 전체 중 처음 3글자를 단계적으로 학습)
n_step = 3
# 입력값 크기. 알파벳에 대한 one-hot 인코딩이므로 26개가 된다
# c => [0 0 1 0 0 ... 0]
# 출력값도 입력값과 마찬가지로 26개의 알파벳으로 분류한다.
n_input = n_class = dic_len

X = tf.placeholder(tf.float32, [None, n_step, n_input])
# 실측값인 Y는 batch_size 에 해당하는 하나의 차원만 있다.(one-hot x. 인덱스 숫자 그대로 사용->[3], [15], [4] ...)
# 기존처럼 one-hot 인코딩을 사용한다면 형태는 [None, n_class] 여야한다.
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# 두 개의 RNN 셀을 생성한다.(여러 셀을 조합해 심층 심경망을 만들기 위해서)
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
# 과적합 방지를 위한 Dropout 기법 적용
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

# MultiRNNCell 함수를 사용하여 여러개의 셀을 조합한 RNN 셀을 생성한다.
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

# dynamic_rnn 함수를 사용하여 심층 순환 싱경망, 즉 Deep RNN 을 만든다.
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

# outputs : [batch_size, n_step, n_hidden] => batch_size : 10(10개 단어), n_step : 3, n_hidden : 128
#        -> [n_step, batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
#        -> [batch_size, n_hidden]
outputs = outputs[-1]

# model : [batch_size, n_class]
model = tf.matmul(outputs, W) + b

# softmax_cross_entropy_with_logits 함수는 label 값을 one-hot 인코딩으로 넘겨줘야 하지만,
# spare_softmax_cross_entropy_with_logits 는 실측값, 즉 labels 값에 one-hot 인코딩을
# 사용하지 않아도 자동으로 변환하여 계산해준다.
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 알파벳의 인덱스로 이루어져 있다.
input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

print('optimization end.')

# 레이블값이 정수이므로 예측값도 정수로 변경한다.
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
# one-hot 인코딩이 아니므로 입력값을 그대로 비교한다.
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})

predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)
