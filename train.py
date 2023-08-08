from process import *
import os
model_file = os.path.join(os.path.dirname(__file__),'models','rainfall_pre_1.h5')

batch_size=128
#（8）模型构建
inputs_shape = sample[0].shape[1:]  # [6,7]  不需要写batch的维度大小
inputs = keras.Input(shape=inputs_shape)  # 输入层

# LSTM层，（设置l2正则化）
#x = layers.LSTM(units=8, dropout=0.5, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
x = layers.LSTM(units=8, return_sequences=True)(inputs)
x = layers.LeakyReLU()(x)
x = layers.LSTM(units=16,  return_sequences=True)(inputs)
x = layers.LeakyReLU()(x)
x = layers.LSTM(units=32)(x)
x = layers.LeakyReLU()(x)
# 全连接层，随即正态分布的权重初始化，（l2正则化）
#x = layers.Dense(64,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.Dense(64,kernel_initializer='random_normal')(x)
# x = layers.Dropout(0.5)(x)
# 输出层返回回归计算后的未来三个时间点的降雨值
outputs = layers.Dense(3)(x)  # 标签shape要和网络shape一样
# 构建模型
model = keras.Model(inputs, outputs)

# 查看网络结构
model.summary()

# 网络编译
model.compile(optimizer=keras.optimizers.Adam(1e-4),  # adam优化器学习率0.001
              loss=tf.keras.losses.MeanAbsoluteError(),metrics=['acc'])  # 计算标签和预测之间绝对差异的平均值

epochs = 15  # 网络迭代次数

# 网络训练
# history = model.fit_generator(train_ds, steps_per_epoch=x_train.shape[0]//batch_size,epochs=epochs, validation_data=val_ds,validation_steps=x_val.shape[0]//batch_size)
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
# 测试集评价
model.evaluate(test_ds)  # loss: 0.1212
#模型保存路径
model.save(model_file)


#（10）查看训练信息
history_dict = history.history  # 获取训练的数据字典
train_loss = history_dict['loss']  # 训练集损失
val_loss = history_dict['val_loss']  # 验证集损失

#（11）绘制训练损失和验证损失，查看训练情况
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')  # 训练集损失
plt.plot(range(epochs), val_loss, label='val_loss')  # 验证集损失
plt.legend()  # 显示标签
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
