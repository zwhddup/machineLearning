## 数据集
> year,month,passengers
1949,January,112
1949,February,118
1949,March,132
1949,April,129
1949,May,121
1949,June,135
1949,July,148
1949,August,148
1949,September,136
1949,October,119
1949,November,104
1949,December,118
1950,January,115
1950,February,126
1950,March,141
1950,April,135
1950,May,125
1950,June,149
1950,July,170
1950,August,170
1950,September,158
1950,October,133
1950,November,114
1950,December,140
1951,January,145
1951,February,150
1951,March,178
1951,April,163
1951,May,172
1951,June,178
1951,July,199
1951,August,199
1951,September,184
1951,October,162
1951,November,146
1951,December,166
1952,January,171
1952,February,180
1952,March,193
1952,April,181
1952,May,183
1952,June,218
1952,July,230
1952,August,242
1952,September,209
1952,October,191
1952,November,172
1952,December,194
1953,January,196
1953,February,196
1953,March,236
1953,April,235
1953,May,229
1953,June,243
1953,July,264
1953,August,272
1953,September,237
1953,October,211
1953,November,180
1953,December,201
1954,January,204
1954,February,188
1954,March,235
1954,April,227
1954,May,234
1954,June,264
1954,July,302
1954,August,293
1954,September,259
1954,October,229
1954,November,203
1954,December,229
1955,January,242
1955,February,233
1955,March,267
1955,April,269
1955,May,270
1955,June,315
1955,July,364
1955,August,347
1955,September,312
1955,October,274
1955,November,237
1955,December,278
1956,January,284
1956,February,277
1956,March,317
1956,April,313
1956,May,318
1956,June,374
1956,July,413
1956,August,405
1956,September,355
1956,October,306
1956,November,271
1956,December,306
1957,January,315
1957,February,301
1957,March,356
1957,April,348
1957,May,355
1957,June,422
1957,July,465
1957,August,467
1957,September,404
1957,October,347
1957,November,305
1957,December,336
1958,January,340
1958,February,318
1958,March,362
1958,April,348
1958,May,363
1958,June,435
1958,July,491
1958,August,505
1958,September,404
1958,October,359
1958,November,310
1958,December,337
1959,January,360
1959,February,342
1959,March,406
1959,April,396
1959,May,420
1959,June,472
1959,July,548
1959,August,559
1959,September,463
1959,October,407
1959,November,362
1959,December,405
1960,January,417
1960,February,391
1960,March,419
1960,April,461
1960,May,472
1960,June,535
1960,July,622
1960,August,606
1960,September,508
1960,October,461
1960,November,390
1960,December,432


```python
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
```

## 训练次数及误差
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210511203630328.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1pXSFNPVUw=,size_16,color_FFFFFF,t_70)

## 训练数据拟合
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210508211820727.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1pXSFNPVUw=,size_16,color_FFFFFF,t_70)

## 测试数据
![在这里插入图片描述](https://img-blog.csdnimg.cn/202105082121573.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1pXSFNPVUw=,size_16,color_FFFFFF,t_70)
## 损失率
每训练1000次打印一次
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210511203907816.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1pXSFNPVUw=,size_16,color_FFFFFF,t_70)

## 参考资料
[理解反向传播](https://blog.csdn.net/weixin_38347387/article/details/82936585)

[理解LSTM](https://www.cnblogs.com/pinard/p/6519110.html)

[LSTM实现1](https://blog.csdn.net/yuechuxuan/article/details/79795503?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-0&spm=1001.2101.3001.4242)

[LSTM实现2](https://blog.csdn.net/PoGeN1/article/details/85137125)
