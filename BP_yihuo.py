import numpy as np
# 输入数据
X = np.array([[1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]
              ])
# 标签
Y = np.array([[0,1,1,0]])
# 初始化权值V，3行4列，取值范围：-1～1
V = np.random.random((3,4))*2 -1
# 初始化权值W，4行1列，取值范围-1～1
W = np.random.random((4,1))*2-1
print(V) # 打印当前权值
print(W)
# 设置学习率
lr = 0.11


def sigmoid(x):
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    return x*(1-x)


def update():
    global X,Y,W,V,lr

    L1 = sigmoid(np.dot(X,V))
    L2 = sigmoid(np.dot(L1,W))

    L2_delta = (Y.T - L2)*dsigmoid(L2)
    L1_delta = L2_delta.dot(W.T)*dsigmoid(L1)

    W_C = lr*L1.T.dot(L2_delta)
    V_C = lr*X.T.dot(L1_delta)

    W += W_C
    V += V_C


for i in range(20000):# 训练2万次
    update()
    if i%500==0:# 每500次显示训练结果
        L1 = sigmoid(np.dot(X,V))
        L2 = sigmoid(np.dot(L1,W))
        print("=========")
        print(L1)
        print(L2)
        print("Error:",np.mean(np.abs(Y.T-L2)))

print(L2)