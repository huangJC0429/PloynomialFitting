import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    s = 1./(1+np.exp(-z))
    return s

temp = np.loadtxt("sin_data.csv", delimiter=',')
# print(temp)
X = temp[:, 0].reshape(1, -1) #x,y都转成n,1的矩阵
Y = temp[:, 1].reshape(1, -1)
n_x = 100
print(n_x)
W1 = np.zeros(shape=(1, 1), dtype = np.float32)
W2 = np.zeros(shape=(1, 1), dtype = np.float32)
W3 = np.zeros(shape=(1, 1), dtype = np.float32)
b = np.zeros(shape=(1, 1), dtype = np.float32)

learning_rate = 0.6
for i in range(10000):
    Z = np.dot(W1, X) + np.dot(W2, X**2) + np.dot(W3, X**3) + b#这里Z的形状是1*n_x
    L = (Z-Y)**2
    dZ = 2*(Z-Y)
    dW1 = 1./n_x*np.dot(X, dZ.T)
    dW2 = 1./n_x*np.dot((X**2), dZ.T)
    dW3 = 1./n_x*np.dot((X**3), dZ.T)
    db = 1./n_x*np.sum(dZ, axis=1)

    W1 = W1 - learning_rate*dW1
    W2 = W2 - learning_rate*dW2
    W3 = W3 - learning_rate*dW3
    b = b - learning_rate*db
    if i % 100 == 0 :
        print("迭代的次数: %i ， 误差值： %f" % (i, 1/n_x*np.sum(L)))

y_hat = W1*X + W2*X**2 + W3*X**3 + b
plt.scatter(X, Y, color='blue')
plt.plot(X, y_hat, 'v')

plt.show()
