
# coding: utf-8

# In[1]:


import os
import numpy as np
X = []
Y = []

for root, dirs, files in os.walk("D:\EarthquakeProjectData\diz"):
    if (("false" in root)&("_normal" in root)):
        print(root)
        earthquakeList = os.listdir(root)
        for i in range(len(earthquakeList)):
            eqPath = os.path.join(root, earthquakeList[i])
            #print(eqPath)
            x = np.array(np.loadtxt(eqPath)).astype(np.float32)
            a = []
            for j in range(3059):
                temp = []
                for k in range(10):
                    temp.append(x[k+j*5])
                avg = np.mean(temp)
                a.append(avg)
            X.append(a)
            if i%3 == 1 :
                Y.append([0,1])


            
for root, dirs, files in os.walk("D:\EarthquakeProjectData\diz"):
    if (("true" in root)&("_normal" in root)):
#         print(root)
        earthquakeList = os.listdir(root)
        for i in range(len(earthquakeList)):
            eqPath = os.path.join(root, earthquakeList[i])
            #print(eqPath)
            x = np.array(np.loadtxt(eqPath)).astype(np.float32)
            a = []
            for j in range(3059):
                temp = []
                for k in range(10):
                    temp.append(x[k+j*5])
                avg = np.mean(temp)
                a.append(avg)
            X.append(a)
            if i%3 == 1:
                Y.append([1,0])
            


# In[6]:


X=np.array(X)
Y=np.array(Y)
size = Y.shape[0]
#size = int(Y.shape[0]/3)

print(X.shape)
print(Y.shape)


# In[7]:


import random
a = list(range(0, size))
random.shuffle(a)
trainset_size = int(round(size * 9 / 10))
testset_size = Y.shape[0] - trainset_size

X = X.reshape(size,3,3059)

x_train = X[a[:trainset_size]].reshape(-1,3059*3)
x_test = X[a[trainset_size:]].reshape(-1,3059*3)

y_train =  Y[a[:trainset_size]]
y_test =  Y[a[trainset_size:]]

print(x_train.shape)
print(testset_size)
print(x_train)


# In[8]:


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

sess = tf.InteractiveSession()  # 创建session


# 一，函数声明部分

def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape,mean=0.0001 ,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def gamma_BN_variable(shape):
    #是batch_normalization方法中的gamma参数，需要初始化之后送到tf.nn.batch_normalization中去
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial)

#added for batch normalization
def beta_BN_variable(shape):
    #BN中的beta参数
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial)

def conv2d(x, W):
    # 参数1：input[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
    # 参数2：filter[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    # 参数3：strides【卷积时在图像每一维的步长，这是一个一维的向量，长度4】
    # 参数4：padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv3d(x,W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 参数1：input
    # 参数2：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]
    # 参数3：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]

    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

def max_pool_4x4(x):
    # 为了加快信息量缩减而设置的较大的池化层
    # 参数信息同上，若网络层数增加则未必会使用（要不可能下降的太快了）
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME')

def max_pool_1x1(x):
    
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='SAME')


# In[7]:

x = tf.placeholder(tf.float32, [None,9177])

# 2个类别，对应输出分类结果
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

# In[8]:


xs = tf.reshape(x, [-1, 1,9177, 1])

# In[9]:


print(ys.shape)
print(x.shape)
print(xs.shape)


# In[9]:


## 第一层卷积操作 ##
# 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
W_conv1 = weight_variable([1, 3, 1, 16])
# 对于每一个卷积核都有一个对应的偏置量。
b_conv1 = bias_variable([16])
# 图片乘以卷积核，并加上偏执量
h_temp1 = conv2d(xs,W_conv1)+b_conv1
gamma_BN1 = gamma_BN_variable([16])
beta_BN1 = beta_BN_variable([16])
h_BN1 = tf.nn.batch_normalization(h_temp1, 0, 0.1, gamma_BN1, beta_BN1, 0.0000001, None)
h_conv1 = tf.nn.relu(h_BN1)
# 池化结果1*7650x32 卷积结果乘以池化卷积核
h_pool1 = max_pool_2x2(h_conv1)
print(W_conv1.shape)


## 第二层卷积操作 ##
# 32通道卷积，卷积出64个特征
W_conv2 = weight_variable([1, 3, 16, 16])
# 64个偏执数据
b_conv2 = bias_variable([16])
# 注意h_pool1是上一层的池化结果
h_temp2 = conv2d(h_pool1, W_conv2) + b_conv2
gamma_BN2 = gamma_BN_variable([16])
beta_BN2 = beta_BN_variable([16])
h_BN2 = tf.nn.batch_normalization(h_temp2, 0, 0.1, gamma_BN2, beta_BN2, 0.0000001, None)
h_conv2 = tf.nn.relu(h_BN2)
# 池化结果1*3825x64
h_pool2 = h_conv2


## 第三层卷积操作 ##
# 32通道卷积，卷积出64个特征
W_conv3 = weight_variable([1, 3, 16, 24])
# 64个偏执数据
b_conv3 = bias_variable([24])
h_temp3 = conv2d(h_pool2, W_conv3) + b_conv3
# 初始化BN中的两个参数gamma和beta
gamma_BN3 = gamma_BN_variable([24])
beta_BN3 = beta_BN_variable([24])
h_BN3 = tf.nn.batch_normalization(h_temp3, 0, 0.1, gamma_BN3, beta_BN3, 0.0000001, None)
h_conv3 = tf.nn.relu(h_BN3)
# 池化结果1*1913*128
h_pool3 =max_pool_2x2(h_conv3)


## 第四层卷积操作 ##
# 32通道卷积，卷积出64个特征
W_conv4 = weight_variable([1, 3, 24, 32])
# 64个偏执数据
b_conv4 = bias_variable([32])
h_temp4 = conv2d(h_pool3, W_conv4) + b_conv4
gamma_BN4 = gamma_BN_variable([32])
beta_BN4 = beta_BN_variable([32])
h_BN4 = tf.nn.batch_normalization(h_temp4, 0, 0.1, gamma_BN4, beta_BN4, 0.0000001, None)
h_conv4 = tf.nn.relu(h_BN4)
# 池化结果1*957x256
h_pool4 = max_pool_2x2(h_conv4)

print(h_pool4.shape)


# In[10]:




## 第五层卷积操作 ##
# 32通道卷积，卷积出64个特征
W_conv5 = weight_variable([1, 3, 32, 32])
# 64个偏执数据
b_conv5 = bias_variable([32])
h_temp5 = conv2d(h_pool4, W_conv5) + b_conv5
# 初始化BN中的两个参数gamma和beta
gamma_BN5 = gamma_BN_variable([32])
beta_BN5 = beta_BN_variable([32])
h_BN5 = tf.nn.batch_normalization(h_temp5, 0, 0.1, gamma_BN5, beta_BN5, 0.0000001, None)
h_conv5 = tf.nn.relu(h_BN5)
# 池化结果1*1913*128
h_pool5 = max_pool_2x2(h_conv5)


## 第六层卷积操作 ##
# 32通道卷积，卷积出64个特征
W_conv6 = weight_variable([1, 3, 32, 48])
# 64个偏执数据
b_conv6 = bias_variable([48])
h_temp6 = conv2d(h_pool5, W_conv6) + b_conv6
# 初始化BN中的两个参数gamma和beta
gamma_BN6 = gamma_BN_variable([48])
beta_BN6 = beta_BN_variable([48])
h_BN6 = tf.nn.batch_normalization(h_temp6, 0, 0.1, gamma_BN6, beta_BN6, 0.0000001, None)
h_conv6 = tf.nn.relu(h_BN6)
# 池化结果1*1913*128
h_pool6 = max_pool_2x2(h_conv6)

## 第七层卷积操作 ##
# 32通道卷积，卷积出64个特征
W_conv7 = weight_variable([1, 3, 48, 64])
# 64个偏执数据
b_conv7 = bias_variable([64])
h_temp7 = conv2d(h_pool6, W_conv7) + b_conv7
# 初始化BN中的两个参数gamma和beta
gamma_BN7 = gamma_BN_variable([64])
beta_BN7 = beta_BN_variable([64])
h_BN7 = tf.nn.batch_normalization(h_temp7, 0, 0.1, gamma_BN7, beta_BN7, 0.0000001, None)
h_conv7 = tf.nn.relu(h_BN7)
# 池化结果1*1913*128
h_pool7 = h_conv7

## 第八层卷积操作 ##
# 32通道卷积，卷积出64个特征
W_conv8 = weight_variable([1, 3, 64, 64])
# 64个偏执数据
b_conv8 = bias_variable([64])
h_temp8 = conv2d(h_pool7, W_conv8) + b_conv8
# 初始化BN中的两个参数gamma和beta
gamma_BN8 = gamma_BN_variable([64])
beta_BN8 = beta_BN_variable([64])
h_BN8 = tf.nn.batch_normalization(h_temp8, 0, 0.1, gamma_BN8, beta_BN8, 0.0000001, None)
h_conv8 = tf.nn.relu(h_BN8)
# 池化结果1*1913*128
h_pool8 = max_pool_2x2(h_conv8)


print(h_pool6.shape)
print(h_pool8.shape)


# In[14]:


## 第一层全连接操作 ##
# 二维张量，第一个参数1*133*64的patch，第二个参数代表卷积个数共1024个
W_fc1 = weight_variable([144*64, 128])
# 1024个偏执数据
b_fc1 = bias_variable([128])
h_pool2_flat = tf.reshape(h_pool8, [-1, 144*64])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
# 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 对卷积结果执行dropout操作（暂时先删去，主要是对0.65这个参数不放心）



## 第二层全连接操作 ##
# 二维张量，给1024这个很大的数进行一次缓冲，以后甚至可以再加入一个全连接层
W_fc2 = weight_variable([128, 84])
# 1024个偏执数据
b_fc2 = bias_variable([84])
h_pool3_flat = tf.reshape(h_fc1_drop, [-1, 128])
h_fc2 = tf.nn.tanh(tf.matmul(h_pool3_flat, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)




# 最终层输出操作 ##
# 二维张量，1*1024矩阵卷积
W_fc3 = weight_variable([84, 2])
b_fc3 = bias_variable([2])
# 最后的分类， softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
# 四，定义loss(最小误差概率)，选定优化优化loss，
cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv)) # 定义交叉熵为loss函数
#cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv + 1e-10))
#cross_entropy = -tf.reduce_mean(ys * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_conv,ys)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.0001, global_step, 2000, 0.96, staircase = True)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step = global_step)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化


# In[17]:


# 开始数据训练以及评测
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()  
for i in range(5000000):
    
    a = list(range(0, trainset_size))
    random.shuffle(a)
    train_size = 256
    batch_x = x_train[a[:train_size]]
    batch_y = y_train[a[:train_size]]
    
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_x, ys: batch_y , keep_prob: 0.65})
        train_entropy = cross_entropy.eval(feed_dict={x: batch_x, ys: batch_y , keep_prob: 0.65})
        current_learning_rate = learning_rate.eval()
        print("step %d, training loss %g, training accuracy %f, with learning rate %f " % (i, train_entropy, train_accuracy, current_learning_rate))
    train_step.run(feed_dict={x: batch_x, ys: batch_y, keep_prob: 0.65})
    if i % 1000==0:
        
        #a = list(range(0, testset_size))
        #random.shuffle(a)
        #train_size = 512
        #batch_x_test = x_train[a[:train_size]]
        #batch_y_test = y_train[a[:train_size]]
        
        print(" ")
        print("test accuracy %f" % accuracy.eval(feed_dict={x: x_test, ys: y_test, keep_prob: 1.0}))
        print(" ")

print(" ")
print("test accuracy %f" % accuracy.eval(feed_dict={x: x_test, ys: y_test, keep_prob: 1.0}))


# In[19]:


print("test accuracy %g" % accuracy.eval(feed_dict={x: x_test, ys: y_test, keep_prob: 1.0}))


# In[18]:


saver=tf.train.Saver()
saver.save(sess, os.path.join(os.getcwd(), 'NO=19_CHG03_variables_batchsize=256_strides=1_witBN_NO=19.ckpt'))


# In[ ]:


# saver=tf.train.Saver()
# saver.save(sess,"Model/newmodel/model.ckpt")


# print("a")
