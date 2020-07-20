'''
wine.txt
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def sigmoid(z):
    s = 1./(1+np.exp(-z))
    return s

def initialparameter(X):
    n_x = X.shape[0]#11
    W = np.zeros((n_x, 1), dtype=np.float32)
    b = 0
    parameter = {"W": W,
                 "b": b}
    return parameter

def model(X,Y,iteration,learning_rate):
    m = X.shape[1]
    parameter = initialparameter(X)
    W = parameter["W"]
    b = parameter["b"]
    costs = []
    for i in range(iteration):
        A = sigmoid(np.dot(W.T, X) + b)  # 1*m
        # 一定要是1.而不是1
        cost = (-1. / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1)
        # 反向传播
        dW = (1. / m) * np.dot(X, (A - Y).T)
        db = (1. / m) * np.sum(A - Y, axis=1)
        # 使用断言确保我的数据是正确的
        assert (dW.shape == W.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)  # squeeze函数的作用是去掉维度为1的维,在这就是将一个一维变成一个数字
        costs.append(cost)
        W = W - learning_rate*dW
        b = b - learning_rate*db

        if i%100 == 0:
            print("以训练"+str(i)+"个数据,损失是"+str(cost))
    parameters = {"W": W,
                  "b": b}
    return parameters, costs

def predict(parameters, X):

    W = parameters["W"]
    b = parameters["b"]
    m = X.shape[1]  # 红酒的数量
    Z = np.zeros((1, m))
    # 计预测
    Z = sigmoid(np.dot(W.T, X) + b)
    assert(Z.shape == (1, m))
    return Z

def usepakage(X, Y, test_X, test_Y):
    Y = Y.reshape(-1, 1)
    X = X.T
    test_Y = test_Y.reshape(-1, 1)
    test_X = test_X.T
    lrModel = LinearRegression()
    lrModel.fit(X, Y)
    #lrModel.predict(test_X, test_Y)
    # 查看参数
    print(lrModel.coef_)
    print(lrModel.intercept_)
    #print(lrModel.score(test_X, test_Y))
    np.set_printoptions(precision=2)  # 设置numpy输出精度为2
    pred_Y = lrModel.predict(test_X)
    print(pred_Y.T)
    print(test_Y.T)

def jiexijie(X, y, test_X, test_Y):
    X = X.T
    y = y.T
    m = X.shape[0]
    # 将两个矩阵组合成一个矩阵。得到的X_b是100行2列的矩阵。其中第一列全都是1.这里填充一个全为1的一列是为了求截距
    X_b = np.c_[np.ones((m, 1)), X]
    print(np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).shape)
    W = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(W)
    # 生成两个新的数据点,得到的是两个x1的值
    X_new = np.array([[0], [2]]).reshape(2,1)
    #X_new_b = np.c_[(np.ones((2, 1))), X_new]
    test_X = test_X.T
    test_Y = test_Y.T
    a = test_X.shape[0]
    #这里填充一个全为1的一列是为了求截距
    test_X = np.hstack(((np.ones((a, 1)).reshape(a, 1)), test_X))
    y_hat = test_X.dot(W)
    np.set_printoptions(precision=2)  # 设置numpy输出精度为2
    print(y_hat.T)
    print(test_Y.T)
    # X_new_b = np.hstack(((np.ones((2, 1)).reshape(2, 1)), X_new))
    #
    # print(X_new_b)
    #
    # # 用求得的theata和构建的预测点X_new_b相乘，得到yhat
    # y_predice = X_new_b.dot(W)
    # print(y_predice)
    # # 画出预测函数的图像，r-表示为用红色的线
    # plt.plot(X_new, y_predice, 'r-', label='jiexijie')

if __name__ == '__main__':
    data = np.loadtxt("wine.txt", delimiter=';')[:, 0:14]
    train_data = data[:-20]
    test_data = data[-20:]
    train_X = train_data[:, 0:11].T/100
    train_Y = train_data[:, 11].reshape(1, -1)/100.
    test_X = test_data[:, 0:11].T/100
    test_Y = test_data[:, 11].reshape(1, -1)/100.


#梯度下降，自由发挥版
    # parameters, costs = model(train_X, train_Y, 50000, 0.01)
    # Z = predict(parameters, test_X)
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations')
    #
    # plt.show()
    # np.set_printoptions(precision=2)#设置numpy输出精度为2
    # print("predict result:"+str(Z))
    # print("   real result:"+str(test_Y))

#调包
    # usepakage(train_X, train_Y, test_X, test_Y)

#解析解
    jiexijie(train_X, train_Y, test_X, test_Y)
