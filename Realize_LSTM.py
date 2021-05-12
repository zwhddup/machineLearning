#实现LSTM
import numpy as np
import numpy
import math
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 激活函数sigmoid和tanh
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):
        return output * (1 - output)


class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0
    def backward(self, output):
        return 1 - output * output

class LSTM(object):
    def __init__(self, input_width, state_width, learning_rate):
        #输入向量的维度
        self.input_width = input_width
        #输出h向量的维度
        self.state_width = state_width
        self.learning_rate = learning_rate

        # 门的激活函数
        self.sigmoid_activator = SigmoidActivator()
        # 输出的激活函数
        self.tanh_activator = TanhActivator()

        # 当前时刻初始化为t0
        self.times = 0

        # 各个时刻的单元状态向量c
        self.ct_list = self.init_state_vec()
        #遗忘门
        self.ft_list = self.init_state_vec()
        #输入门
        self.it_list = self.init_state_vec()
        self.at_list = self.init_state_vec()
        # 细胞更新

        #输出门
        self.ot_list = self.init_state_vec()
        self.ht_list = self.init_state_vec()

        #各个门的权重矩阵
        #遗忘门
        self.wf, self.uf, self.bf = self.init_weight_mat()
        #输入门
        self.wi, self.ui, self.bi = self.init_weight_mat()
        self.wa, self.ua, self.ba = self.init_weight_mat()
        #输出门
        self.wo, self.uo, self.bo = self.init_weight_mat()


    #lstm前向传播
    def forward(self, x):
        #更新时间
        self.times += 1
        #遗忘门
        ft = self.calc_gate(x, self.wf, self.uf, self.bf, self.sigmoid_activator)
        self.ft_list.append(ft)
        #输入门
        it = self.calc_gate(x, self.wi, self.ui, self.bi, self.sigmoid_activator)
        self.it_list.append(it)
        at = self.calc_gate(x, self.wa, self.ua, self.ba, self.tanh_activator)
        self.at_list.append(at)
        #更新细胞状态
        ct = self.ct_list[self.times - 1] * ft + it * at
        self.ct_list.append(ct)
        #输出门
        ot = self.calc_gate(x, self.wo, self.uo, self.bo, self.sigmoid_activator)
        self.ot_list.append(ot)
        ht = ot * self.tanh_activator.forward(ct)
        self.ht_list.append(ht)



    # 初始化权重矩阵
    def init_weight_mat(self):
        Wh = np.random.uniform(-1e-4, 1e-4, (self.state_width, self.state_width))
        Wx = np.random.uniform(-1e-4, 1e-4, (self.state_width, self.input_width))
        b = np.zeros((self.state_width, 1))
        return Wh, Wx, b

    # 初始化保存状态的向量为0
    def init_state_vec(self):
        state_vec_list = []
        state_vec_list.append(np.zeros((self.state_width, 1)))
        return state_vec_list

    # 计算门
    def calc_gate(self, x, Wh, Wx, b, activator):
        # 上次的LSTM输出
        h = self.ht_list[self.times - 1]
        net = np.dot(Wh, h) + np.dot(Wx, x) + b
        gate = activator.forward(net)
        return gate

    #反向传播
    def backward(self, x, delta_h):
        #计算误差项
        self.calc_delta(delta_h)
        #计算梯度
        self.calc_gradient(x)
        #更新权重
        self.update()

    def print_par(self):
        print("------------par-------------")
        print(self.ft_list)
        print(self.it_list)
        print(self.at_list)
        print(self.ot_list)
        print(self.ht_list)
        print(self.ct_list)
        print("------------weight-------------")
        print(self.wf, self.uf, self.bf)
        print(self.wo, self.uo, self.bo)
        print(self.wa, self.ua, self.ba)
        print(self.wi, self.ui, self.bi)

    def update(self):
        #print("--------update weight---------")
        self.wf -= self.learning_rate * self.wf_grad
        self.uf -= self.learning_rate * self.uf_grad
        self.bf -= self.learning_rate * self.bf_grad

        self.wi -= self.learning_rate * self.wi_grad
        self.ui -= self.learning_rate * self.ui_grad
        self.bi -= self.learning_rate * self.bi_grad

        self.wo -= self.learning_rate * self.wo_grad
        self.uo -= self.learning_rate * self.uo_grad
        self.bo -= self.learning_rate * self.bo_grad

        self.wa -= self.learning_rate * self.wa_grad
        self.ua -= self.learning_rate * self.ua_grad
        self.ba -= self.learning_rate * self.ba_grad

    #计算各个时刻的误差项
    #delta_h为所有时刻 实际值-预测值 的误差
    def calc_delta(self, delta_h):

        #输出门误差项
        self.delta_o_list = self.init_delta()
        #输入门误差项
        self.delta_i_list = self.init_delta()
        #遗忘门误差项
        self.delta_f_list = self.init_delta()
        #即时状态at的误差项
        self.delta_a_list = self.init_delta()

        self.delta_h_list = delta_h

        #计算每个时刻的误差项
        for k in range(self.times, 0, -1):
            self.calc_delta_k(k)



    #计算k时刻的误差项
    def calc_delta_k(self, k):
        #获取各个前项计算值
        i = self.it_list[k]
        o = self.ot_list[k]
        f = self.ft_list[k]
        a = self.at_list[k]

        c = self.ct_list[k]
        c_pre = self.ct_list[k-1]
        tanh_c = self.tanh_activator.forward(c)

        delta_k = self.delta_h_list[k]

        #计算delta
        delta_o = (delta_k * tanh_c * self.sigmoid_activator.backward(o))
        self.delta_o_list[k] = delta_o
        delta_f = (delta_k * o * self.tanh_activator.backward(tanh_c) * c_pre * self.sigmoid_activator.backward(f))
        self.delta_f_list[k] = delta_f
        delta_i = (delta_k * o * self.tanh_activator.backward(tanh_c) * a * self.sigmoid_activator.backward(i))
        self.delta_i_list[k] = delta_i
        delta_a = (delta_k * o * self.tanh_activator.backward(tanh_c) * i * self.tanh_activator.backward(a))
        self.delta_a_list[k] = delta_a


    #计算梯度
    def calc_gradient(self, x):
        self.wf_grad, self.uf_grad, self.bf_grad = (self.init_weight_mat())
        self.wi_grad, self.ui_grad, self.bi_grad = (self.init_weight_mat())
        self.wa_grad, self.ua_grad, self.ba_grad = (self.init_weight_mat())
        self.wo_grad, self.uo_grad, self.bo_grad = (self.init_weight_mat())

        for t in range(self.times, 0, -1):
            (wf_grad, uf_grad, bf_grad,
             wi_grad, ui_grad, bi_grad,
             wo_grad, uo_grad, bo_grad,
             wa_grad, ua_grad, ba_grad) = (self.calc_gradient_k(t, x))
            #实际梯度等于各梯度之和
            self.wf_grad += wf_grad
            self.uf_grad += uf_grad
            self.bf_grad += bf_grad


            self.wi_grad += wi_grad
            self.ui_grad += ui_grad
            self.bi_grad += bi_grad

            self.wo_grad += wo_grad
            self.uo_grad += uo_grad
            self.bo_grad += bo_grad

            self.wa_grad += wa_grad
            self.ua_grad += ua_grad
            self.ba_grad += ba_grad

    #计算各个时刻t的w, u, b权重梯度
    def calc_gradient_k(self, k, x):
        h_prev = self.ht_list[k-1].transpose()
        wf_grad = np.dot(self.delta_f_list[k], h_prev)
        uf_grad = np.dot(self.delta_f_list[k], x[k-1].transpose())
        bf_grad = self.delta_f_list[k]

        wi_grad = np.dot(self.delta_i_list[k], h_prev)
        ui_grad = np.dot(self.delta_i_list[k], x[k-1].transpose())
        bi_grad = self.delta_i_list[k]

        wo_grad = np.dot(self.delta_o_list[k], h_prev)
        uo_grad = np.dot(self.delta_o_list[k], x[k-1].transpose())
        bo_grad = self.delta_o_list[k]

        wa_grad = np.dot(self.delta_a_list[k], h_prev)
        ua_grad = np.dot(self.delta_a_list[k], x[k-1].transpose())
        ba_grad = self.delta_a_list[k]

        return wf_grad, uf_grad, bf_grad, wi_grad, ui_grad, bi_grad, wo_grad, uo_grad, bo_grad, wa_grad, ua_grad, ba_grad

    #初始化误差项list
    def init_delta(self):
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(np.zeros((self.state_width, 1)))
        return delta_list

    def predict(self, x):
        self.reset_state()
        for i in range(len(x)):
            self.forward(x[i])
        return self.ht_list[1:]

    #重置为0
    def reset_state(self):
        # 当前时刻初始化为t0
        self.times = 0
        # 各个时刻的单元状态向量c
        self.ct_list = self.init_state_vec()
        # 各个时刻的输出向量h
        self.ht_list = self.init_state_vec()
        # 各个时刻的遗忘门f
        self.ft_list = self.init_state_vec()
        # 各个时刻的输入门i
        self.it_list = self.init_state_vec()
        # 各个时刻的输出门o
        self.ot_list = self.init_state_vec()
        # 各个时刻的即时状态at
        self.at_list = self.init_state_vec()


    #损失函数
    def loss(self, y):
        delta_h_list = []
        delta_h_list.append(np.zeros((y.shape[1], 1)))
        for i in range(self.times):
            delta_h_list.append(self.ht_list[i+1] - y[i])
        return delta_h_list

    def train(self, times, train_data, mark_data):
        self.loss_list = []
        t = 1;
        count = 0
        # 训练
        while times > 0:
            for i in range(len(train_data)):
                l.forward(train_data[i])
            # 调参
            temp = l.loss(mark_data)
            l.backward(train_data, temp)
            #计算总的loss
            sum = np.zeros((1, 1))
            for i in temp:
                sum += math.sqrt(math.pow(i,2))
            sum = sum / len(train_data)
            count += 1
            #每训练1000次打印loss
            if count == 1000:
                print("epoch %d: loss: %0.8f" % (t * count, sum[0, 0]))
                self.loss_list.append(sum[0, 0])
                t += 1
                count = 0
            times -= 1
            if times > 0:
                self.reset_state()



def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in dataset[:-look_back]:
        dataX.append(numpy.array([i]))
    for i in dataset[look_back:]:
        dataY.append(i)
    return dataX, numpy.array(dataY)

dataframe = pd.read_csv('flights.csv', usecols=[2], engine='python')
dataset = dataframe.values
# 将整型变为float
dataset = dataset.astype('float32')
#plt.plot(dataset)
#plt.show()
#归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print("train: " + str(len(train)) + "\ntest: " + str(len(test)))
# 前面的全部数据，预测后一个月的数据
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

l = LSTM(1, 1, 0.001)
l.train(100000, trainX, trainY)
#归一化逆转
#print(l.ht_list)
ans = []
for i in l.ht_list[1:]:
    ans.append(i[0])
ans = np.array(ans)
#print(len(ans))
#print(ans)
ans = scaler.inverse_transform(ans)
trainY = scaler.inverse_transform(trainY)
#print(ans)
plt.plot(trainY)
plt.plot(ans)
plt.show()

ans = []
for i in l.predict(testX):
    ans.append(i[0])
ans = np.array(ans)
ans = scaler.inverse_transform(ans)
testY = scaler.inverse_transform(testY)

plt.plot(testY)
plt.plot(ans)
plt.show()

plt.plot(l.loss_list)
plt.show()














def test():
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()

    x = [np.array([[1], [2], [3]]),
         np.array([[5], [3], [4]]),
         np.array([[7], [9], [11]])]
    d = np.array([[5],
                  [3],
                  [6]])
    d = d.astype('float32')

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    d = scaler.fit_transform(d)
    #print(d)
    #print(scaler.inverse_transform(d))

    l = LSTM(3, 1, 0.006)
    times = 100000;
    while times > 0:
        l.forward(x[0])
        l.forward(x[1])
        l.forward(x[2])
        l.backward(x, l.loss(d))
        # print(l.ht_list)
        # print(l.ht_list[3])
        temp = l.loss(d)
        #print(temp)
        #计算均方误差
        sum = np.zeros((1, 1))
        for i in temp:
            sum += i
        sum = sum / len(x)
        print("loss: %0.8f" % sum[0, 0])

        times -= 1
        if times > 0:
            l.reset_state()
        #l.print_par()
    #print(l.ht_list)
    #归一化逆转
    ans = []
    for i in l.ht_list[1:]:
        ans.append(i[0])
    ans = np.array(ans)
    #print(ans)
    print(scaler.inverse_transform(ans))
#test()

