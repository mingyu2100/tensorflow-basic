import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mp_image


"""그래프 그리기"""
x = range(-10, 10)
y = [np.log(v) for v in x]
y2 = [np.log(1 - v) for v in x]

plt.figure()
plt.subplot(211)
plt.plot(x, y, label='log(x)')
plt.grid()
plt.legend()

plt.subplot(212)
plt.plot(x, y2, label='log(1 - x)')
plt.grid()
plt.legend()

plt.show()


"""이미지 보여주기"""
filename = 'packt.JPG'
input_image = mp_image.imread(filename)

print('input dim = {}'.format(input_image.ndim))
print('input shape = {}'.format(input_image.shape))

plt.imshow(input_image)
plt.show()
