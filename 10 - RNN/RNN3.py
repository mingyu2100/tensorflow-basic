# Sequence to Sequence(구글이 기계번역에 사용하는 신경망 모델, Seq2Seq)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력의 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
# ex) 현재 배치 데이터의 최대 크기가 4 인 경우
#   word -> ['w', 'o', 'r', 'd']
#   to   -> ['t', 'o', 'P', 'P']
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값을 위해 입력 단어를 한 글자씩 떼어 배열로 만든다.
        input = [num_dic[n] for n in seq[0]]

        # 디코더 셀의 입력값을 위해 출력 단어의 글자들을 배열로 만들고,
        # 시작을 나타내는 심볼 'S'를 맨 앞에 붙인다.
        output = [num_dic[n] for n in ('S' + seq[1])]

        # 학습을 위해 비교할 디코더 셀의 출력값을 만들고,
        # 출력의 끝을 알려주는 심볼 'E'를 마지막에 붙인다.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아니다.(sparese_softamx_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch


learning_rate = 0.01
n_hidden = 128
total_epoch = 100
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_class = n_input = dic_len

# 인코더와 디코더의 입력값들은 원-핫 인코딩을 사용([batch size, time steps, input size]) - time steps : 단어 길이
# 디코더의 출력값은 인덱스 숫자를 그대로 사용하기 때문에 입력값의 랭크(차원)가 하나 더 높다.([batch size, time steps])
# 입력 단계는 배치 크기처럼 입력받을 때마다 다를 수 있으므로 None 으로 설정한다.
# 단, 같은 배치 때 입력되는 데이터는 글자 수, 즉 단계(time steps)가 모두 같아야 한다.
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])
"""
dynamic_rnn 의 옵션인 sequence_length 를 사용하면 길이가 다른 단어들도 한 번에 입력받을 수 있지만,
그래도 입력 데이터의 길이는 같아야 한다.
따라서 입력할 때 짧은 단어는 가장 긴 단어에 맞춰 글자를 채워야 한다.
의미 없는 값인 'P'는 이렇게 부족한 글자 수를 채우는데 사용한다.
이 방식을 사용하려면 코드가 조금 복잡해지기 때문에, 여기서는 길이가 같은 단어만 사용하도록 한다.
"""

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # 디코더를 만들 때 초기 상태값(입력값이 아님)으로 인코더의 최정 상태 값을 넣어줘야 한다.
    # Sequence to Sequence 의 핵심 아이디어 중 하나가 인코더에서 계산한 상태를 디코더로 전파하는 것이다.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)

# 가중치와 편향을 위한 변수를 사용하지 않고, 고수준 API 를 사용하여 귀찮은 부분을 제거한다.
model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={enc_input: input_batch,
                                                     dec_input: output_batch,
                                                     targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.6f}'.format(loss))

print('Optimization end.')


# 결과를 확인하기 위해 단어를 입력받아 번역 단어를 예측하는 함수 작성
# 이 모델은 입력값과 출력값 데이터로 [영어 단어, 한글 단어]를 사용하지만,
# 예측 시에는 한글 단어를 알지 못한다.
# 따라서 디코더의 입출력을 의미 없는 값인 'P'로 채워 데이터를 구성한다.
def translate(word):
    # 입력으로 'word'를 받았다면, seq_data 는 ['word', 'PPPP']로 구성된다.
    seq_data = [word, 'P' * len(word)]

    # input_batch = ['w', 'o', 'r', 'd'], output_batch = ['P', 'P', 'P', 'P'] 글자들의 인덱스를 원-핫 인코딩한 값
    # target_batch = ['P', 'P', 'P', 'P'] 각 글자의 인덱스인 [2, 2, 2, 2]가 된다.
    input_batch, output_batch, target_batch = make_batch([seq_data])

    # 세 번째 차원을 argmax 로 취해 가장 확률이 높은 글자(의 인덱스)를 예측값으로 만든다.
    # 세 번째 차원을 argmax 로 취하는 이유는 결과값이 [batch size, time steps, input size] 형태로 나오기 때문이다.
    # ex. [[[0 0 0.9 0.1 0.2 0.3 0 0 ...] [0 0.1 0.3 0.7 0.1 0 0 0 ...] ...]]
    # tf.argmax(model, 2) => [[[2], [3] ...]]
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction, feed_dict={enc_input: input_batch,
                                             dec_input: output_batch,
                                             targets: target_batch})

    # 예측 결과는 글자의 인덱스를 뜻하는 숫자이므로 각 숫자에 해당하는 글자를 가져와 배열을 만든다.
    decoded = [char_arr[i] for i in result[0]]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    # 이유는 디코더의 입력(time steps) 크기만큼 출력값이 나오므로 최종 결과는 ['사', '랑', 'E', 'E']처럼 나오기 때문이다.
    end = decoded.index('E')
    translate = ''.join(decoded[:end])

    return translate

print('\n=== 번역 테스트 ===')

print('word ->', translate('word'))
print('wodr ->', translate('wodr'))
print('love ->', translate('love'))
print('loev ->', translate('loev'))
print('abcd ->', translate('abcd'))

