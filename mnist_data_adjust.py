
# coding: utf-8

# In[181]:


import pandas as pd
import numpy as np
import tensorflow as tf 


# In[182]:


train=pd.read_csv('./train.csv')
test=pd.read_csv('./test.csv')


# In[183]:


print('train shape is {}'.format(train.shape))
print('test shape is {}'.format(test.shape))


# In[184]:


test.head()


# In[185]:


train.isnull().any().sum()


# In[186]:


train.describe()


# In[187]:


import matplotlib.pyplot as plt
random_index=np.random.randint(0,42000,size=20)
plt.figure(figsize=(12,10))
for i,index in enumerate(random_index):
    plt.subplot(2,10,i+1)
    plt.imshow(train.iloc[index,1:].values.reshape(28,28),cmap='gray')
plt.show()


# In[188]:


t=train['label'].value_counts()
plt.figure(figsize=(12,5))
plt.bar(t.index,t.values,color='green')
plt.xlabel('digits')
plt.ylabel('digits counts')
plt.title('distribution of digits')


# In[189]:


from sklearn.model_selection import train_test_split
seed=16
np.random.seed(seed)
X=train.iloc[:,1:]
Y=train.iloc[:,0]
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=seed)


# In[190]:


train_x=train_x.values.reshape(-1,28,28,1)
test_x=test_x.values.reshape(-1,28,28,1)
train_x=train_x/255.0
test_x=test_x/255.0


# In[191]:


from sklearn.preprocessing import OneHotEncoder
le=OneHotEncoder(sparse=False)
train_y=le.fit_transform(train_y.values.reshape(-1,1))
test_y=le.fit_transform(test_y.values.reshape(-1,1))


# In[192]:


print('train_y shape is {}'.format(train_y.shape))


# # 1.随机裁剪
# 使用tf.random_crop随机裁剪原图片中的一部分
# 使用tf.image.resize_images将图片重塑至原来的大小

# In[193]:


sess=tf.Session()
adjust_data=tf.placeholder(tf.float32,[None,28,28,1])
m=tf.shape(adjust_data)[0]
original_size=[28,28]
crop_size=[m,25,25,1]
seed=10
crop_x=tf.random_crop(adjust_data,size=crop_size,seed=seed)
X_random_crop=tf.image.resize_images(crop_x,[28,28])


# In[194]:


plt.imshow(train_x[1,:,:,0],cmap='gray')


# In[195]:


print(train_x[[1000],:,:,:].shape)
print(train_x[1000,:,:,:].shape)


# In[196]:


plt.imshow(sess.run(X_random_crop,feed_dict={adjust_data:train_x[[1],:,:,:]})[0,:,:,0],cmap='gray')


# # 2.水平、垂直移动
# 使用tf.image.pad_to_bounding_box向图片四周填充3个为0的像素点
# 使用tf.image.crop_to_bounding_box截取图片的指定位置，实现图片的水平和垂直移动

# In[197]:


#原size为28*28，填充后34*34
pad_x=tf.image.pad_to_bounding_box(adjust_data,3,3,34,34)


# In[198]:


up_x=tf.image.crop_to_bounding_box(pad_x,6,3,28,28)
down_x=tf.image.crop_to_bounding_box(pad_x,0,3,28,28)
left_x=tf.image.crop_to_bounding_box(pad_x,3,6,28,28)
right_x=tf.image.crop_to_bounding_box(pad_x,3,0,28,28)

right_down_x=tf.image.crop_to_bounding_box(pad_x,0,0,28,28)
right_up_x=tf.image.crop_to_bounding_box(pad_x,0,6,28,28)
left_down_x=tf.image.crop_to_bounding_box(pad_x,6,0,28,28)
left_up_x=tf.image.crop_to_bounding_box(pad_x,6,6,28,28)


# 移动前原图

# In[199]:


plt.imshow(train_x[1,:,:,0],cmap='gray')


# 左移后

# In[200]:


plt.imshow(sess.run(tf.reshape(left_x,[28,28]),feed_dict={adjust_data:train_x[[1],:,:,:]}),cmap='gray')


# 右移后

# In[201]:


plt.imshow(sess.run(tf.reshape(right_x,[28,28]),feed_dict={adjust_data:train_x[[1],:,:,:]}),cmap='gray')


# 上移后

# In[202]:


plt.imshow(sess.run(tf.reshape(up_x,[28,28]),feed_dict={adjust_data:train_x[[1],:,:,:]}),cmap='gray')


# 下移后

# In[203]:


plt.imshow(sess.run(tf.reshape(down_x,[28,28]),feed_dict={adjust_data:train_x[[1],:,:,:]}),cmap='gray')


# 左上移动

# In[204]:


plt.imshow(sess.run(tf.reshape(left_up_x,[28,28]),feed_dict={adjust_data:train_x[[1],:,:,:]}),cmap='gray')
#plt.imshow(sess.run(tf.reshape(left_down_x,[28,28]),feed_dict={adjust_data:train_x[[1],:,:,:]}),cmap='gray')


# # 3.旋转
#     使用tf.contrib.image.rotate将图片进行旋转至指定角度
#     旋转后可能会出现图像的维数与原来不一致，使用tf.contrib.image.rotate中可以使用最临近值进行插补缺失维度

# In[205]:


forward_x_10=tf.contrib.image.rotate(adjust_data,10*np.pi/180)
backword_x_10=tf.contrib.image.rotate(adjust_data,-10*np.pi/180)
forward_x_15=tf.contrib.image.rotate(adjust_data,15*np.pi/180)
backword_x_15=tf.contrib.image.rotate(adjust_data,-15*np.pi/180)


# In[206]:


plt.imshow(sess.run(tf.reshape(forward_x_10,[28,28]),feed_dict={adjust_data:train_x[[1],:,:,:]}),cmap='gray')


# # 4.数据整合，将翻转，平移后的数据整合到一起

# In[207]:


augment_set=tf.concat([X_random_crop,up_x,down_x,left_x,right_x,left_down_x,left_up_x,right_down_x,right_up_x,forward_x_10,backword_x_10
                      ,forward_x_15,backword_x_15,adjust_data],0)


# In[208]:


train_X=sess.run(augment_set,feed_dict={adjust_data:train_x})


# In[209]:


train_Y=np.tile(train_y,[14,1])


# # 5.数据分割

# In[210]:


def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
    m=X.shape[0]
    mini_batch=[]
    np.random.seed(seed)
    #step 1 shuffle(X,Y)
    permutation=list(np.random.permutation(m))
    shuffle_x=X[permutation,:,:,:]
    shuffle_y=Y[permutation,:]
    num_minibatches=int(m/mini_batch_size)
    for k in range(num_minibatches):
        mini_batch_X=shuffle_x[k*mini_batch_size:k*mini_batch_size+mini_batch_size,:,:,:]
        mini_batch_Y=shuffle_y[k*mini_batch_size:k*mini_batch_size+mini_batch_size,:]
        minibatch=(mini_batch_X,mini_batch_Y)
        mini_batch.append(minibatch)
    if m%mini_batch_size!=0:
        mini_batch_X=shuffle_x[num_minibatches*mini_batch_size:m,:,:,:]
        mini_batch_Y=shuffle_y[num_minibatches*mini_batch_size:m,:]
        minibatch=(mini_batch_X,mini_batch_Y)
        mini_batch.append(minibatch)
    return mini_batch


# # 5.搭建神经网络进行学习

# In[211]:


def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))


# In[212]:


xs=tf.placeholder(tf.float32,shape=[None,28,28,1],name='x')
ys=tf.placeholder(tf.float32,shape=[None,10],name='y_')
#第一层卷积
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(xs,w_conv1)+b_conv1)
pool_1=max_pool(h_conv1)

#第二层卷积
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(pool_1,w_conv2)+b_conv2)
pool_2=max_pool(h_conv2)

#全连接层
w_fc_1=weight_variable([7*7*64,512])
b_fc_1=bias_variable([512])

#输出层
pool_2_flat=tf.reshape(pool_2,[-1,7*7*64])
h_fc_1=tf.nn.relu(tf.matmul(pool_2_flat,w_fc_1)+b_fc_1)
keep_prob=tf.placeholder(tf.float32,name='keep_prob')
h_fc_prob=tf.nn.dropout(h_fc_1,keep_prob)
w_fc_2=weight_variable([512,10])
b_fc_2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc_prob,w_fc_2)+b_fc_2)

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=y_conv))
cross_entropy=-tf.reduce_sum(ys*tf.log(y_conv))
train=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[217]:


init=tf.global_variables_initializer()
sess.run(init)


# In[218]:


trainloss,testloss=[],[]
trainaccuracy,testaccuracy=[],[]
num=int(train_X.shape[0]/64)
for i in range(5):
    seed+=1
    minbatches=random_mini_batches(train_X,train_Y,64,seed)
    epoch_loss=0
    epoch_accuracy=0
    for minbatch in minbatches:
        batch_X,batch_Y=minbatch
        feed_dict={xs:batch_X,ys:batch_Y,keep_prob:1.0}
        _=sess.run(train,feed_dict=feed_dict)
        batch_loss,batch_accuracy=sess.run([cross_entropy,accuracy],feed_dict=feed_dict)
        epoch_loss += batch_loss/num
        epoch_accuracy += batch_accuracy/num      
    test_cost,test_accuracy=sess.run([cross_entropy,accuracy],feed_dict={xs:test_x,ys:test_y,keep_prob:1.0})
    trainloss.append(epoch_loss)
    trainaccuracy.append(epoch_accuracy)
    testloss.append(test_cost)
    testaccuracy.append(test_accuracy)
    if i%1==0:
        print('{}th iter,train loss is {:.5f},train accuracy is {:.5f},test cost is {:.5f},test accuracy is {:.5f}'
              .format(i,epoch_loss,epoch_accuracy,test_cost,test_accuracy))


# In[234]:


plt.subplot(211)
plt.plot(list(range(5)),trainloss,label='train loss')
plt.plot(np.arange(5),testloss,label='test loss')
legend=plt.legend()
plt.subplot(212)
plt.plot(np.arange(5),trainaccuracy,label='train accuracy')
plt.plot(np.arange(5),testaccuracy,label='test accuracy')
legend=plt.legend()


# In[223]:


from sklearn.metrics import confusion_matrix
pred=tf.argmax(y_conv,1)
y_pred=sess.run(pred,feed_dict={xs:test_x,ys:test_y,keep_prob:1.0})
confusion_matrix=confusion_matrix(np.argmax(test_y,1),y_pred)


# In[233]:


plt.imshow(confusion_matrix,cmap=plt.cm.Blues)
plt.xticks(range(10),range(10))
plt.yticks(range(10),range(10))
plt.xlabel('predict results')
plt.ylabel('True results')
for i in range (10):
    for j in range(10):
        plt.text(j,i,confusion_matrix[i,j],horizontalalignment="center")

