import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 调用GPU加速
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


#读取数据集
filepath = 'D:/文件/pycharm-project/rainfall/rainfall.csv'
data = pd.read_csv(filepath)
# print(data.head())  # 数据是10min记录一次的

#特征选择
# 选择从第1列开始往后的所有行的数据
feat = data.iloc[:, 1:8]  # 除时间列其他为特征
date = data.iloc[:, 0]   # 获取时间信息

#利用pandas绘图展示每个特征点分布情况
# feat.plot(subplots=True, figsize=(80,10),  # 为每一列单独开辟子图，设置画板大小
#           layout=(5,2), title='rainfall features')  # 7张图的排序方式，设置标题
# plt.show()

#特征数据预处理
train_num = 20000  # 取前2w组数据用于训练
val_num = 23000  # 取2w-2.3w的数据用于验证
# 2.3w-2.5w的数据用于验证用于测试

# 保存所有的降雨数据，即标签数据
targets = feat.iloc[:,6]
scaler_targets = MinMaxScaler(feature_range=(0,1))
targets_scaler= scaler_targets.fit_transform(targets.values.reshape(-1,1))  # 取归一化之后的降雨强度数据作为标签值


# 特征值归一化
scaler_feature = MinMaxScaler(feature_range=(0,1))
feat_scaler= scaler_feature.fit_transform(feat)


#定义切片函数（自定义预测时间点）
'''
dataset 代表特征数据
start_index 代表从数据的第几个索引值开始取
history_size 滑动窗口大小
end_index 代表数据取到哪个索引就结束
target_size 代表预测未来某一时间点还是时间段的降雨强度。例如target_size=0代表用前20个特征预测第21个的降雨强度
step 代表在滑动窗口中每隔多少步取一组特征
point_time 布尔类型，用来表示预测未来某一时间点的降雨强度，还是时间段的降雨强度
true 原始降雨数据的所有标签值
'''
def TimeSeries(dataset, start_index, history_size, end_index, step,
               target_size, point_time, true):
    data = []  # 保存特征数据
    labels = []  # 保存特征数据对应的标签值

    start_index = start_index + history_size  # 第一次的取值范围[0:start_index]
    # 如果没有指定滑动窗口取到哪个结束，那就取到最后
    if end_index is None:
        # 数据集最后一块是用来作为标签值的，特征不能取到底
        end_index = len(dataset) - target_size

    # 滑动窗口的起始位置到终止位置每次移动一步
    for i in range(start_index, end_index):

        # 滑窗中的值不全部取出来用，每隔60min取一次
        index = range(i - history_size, i, step)  # 第一次相当于range(0, start_index, 6)

        # 根据索引取出所有的特征数据的指定行
        data.append(dataset[index])

        # 用这些特征来预测某一个时间点的值还是未来某一时间段的值
        if point_time is True:  # 预测某一个时间点
            # 预测未来哪个时间点的数据，例如[0:20]的特征数据（20取不到），来预测第20个的标签值
            labels.append(true[i + target_size])

        else:  # 预测未来某一时间区间
            # 例如[0:20]的特征数据（20取不到），来预测[20,20+target_size]数据区间的标签值
            labels.append(true[i:i + target_size])

    # 返回划分好了的时间序列特征及其对应的标签值
    return np.array(data), np.array(labels)

#（6）划分数据集
history_size = 6  # 每个滑窗取1小时的数据量=6
target_size =  3  # 预测未来下三个10分钟时间点的降雨强度
step = 1  # 步长为1取所有的行

# 构造训练集
x_train, y_train = TimeSeries(dataset=feat_scaler, start_index=0, history_size=history_size, end_index=train_num,
                              step=step, target_size=target_size, point_time=False, true=targets_scaler)

# 构造验证集
x_val, y_val = TimeSeries(dataset=feat_scaler, start_index=train_num, history_size=history_size, end_index=val_num,
                          step=step, target_size=target_size, point_time=False, true=targets_scaler)

# 构造测试集
x_test, y_test =  TimeSeries(dataset=feat_scaler, start_index=val_num, history_size=history_size, end_index=25000,
                              step=step, target_size=target_size, point_time=False, true=targets_scaler)


# 查看数据集信息
# print('x_train_shape:', x_train.shape)  # (19994, 6, 7)
# print('y_train_shape:', y_train.shape)  # (19994,)



#构造tf数据集
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # 训练集
train_ds = train_ds.batch(128).shuffle(10000)  # 随机打乱、每个step处理128组数据
# train_ds = train_ds.batch(512)  # 随机打乱、每个step处理128组数据

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))  # 验证集
val_ds = val_ds.batch(128)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 测试集
test_ds = test_ds.batch(128)

# 查看数据集信息

sample = next(iter(train_ds))  # 取出一个batch的数据
# print('x_train.shape:', sample[0].shape)  # [128, 6, 7]
# print('y_train.shape:', sample[1].shape)  # [128,3]
# print('input_shape:', sample[0].shape[-2:])#[6,7]
# print(type(sample[0]))
# print(len(x_train))
# print(sample[0].shape)
# print(sample[1].shape)
# print(x_test.shape)#(1994, 6, 7)
# print(y_test.shape)#(1994, 3, 1)
# print(type(y_test))