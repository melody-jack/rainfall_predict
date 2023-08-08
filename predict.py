from process import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,load_model
import os

#模型保存路径
model_file = os.path.join(os.path.dirname(__file__),'models','rainfall_pre_1.h5')
#固定参数设置
history_size = 6  # 每个滑窗取1小时的数据量=6
target_size =  3  # 预测未来下三个10分钟时间点的降雨强度
step = 1  # 步长为1取所有的行

def predict(model,file):
    """file为与当前脚本同路径的csv文件（包含前6个时间间隔点的特征数据,shape为【7,7】第一列数据为时间，后面为特征）"""
    data_pre = pd.read_csv(file)
    feat_pre = data_pre.iloc[:, 1:8]  # 除时间列其他为特征
    scaler = MinMaxScaler(feature_range=(0, 1))
    feat_scaler_pre = scaler.fit_transform(feat_pre)
    y_predict=model.predict(feat_scaler_pre)
    return  y_predict

if __name__=="__main__":
    """模型加载"""
    model = load_model(model_file)
    """测试数据测试(转化成原始数据)"""
    # x_predict = x_test[:200]  # 用测试集的前200组特征数据来预测
    # y_true = y_test[:200]  # 每组特征对应的标签值
    #
    # y_predict = model.predict(x_predict)  # 对测试集的特征预测

    # 输入为时间间隔为10分钟1小时内的所有特征数据，shape[6,7]
    # x_predict=x_test[100:101]
    # y_true = y_test[100:101][0,:,:]
    # # y_true = y_test[100:101]
    # y_predict = model.predict(x_predict).reshape(-1,1)
    # y_pre=scaler_targets.inverse_transform(y_predict)
    # y_tru=scaler_targets.inverse_transform(y_true)
    # print(x_predict.shape)#(1,6,7)
    # print(y_predict)

    """测试集批量预测"""
    # （12）预测阶段
    # x_test[0].shape = (720,10)
    x_predict = x_test[:200]  # 用测试集的前200组特征数据来预测
    y_true = y_test[:200][:,:,0]  # 每组特征对应的标签值

    y_predict = model.predict(x_predict)  # 对测试集的特征预测
    print(y_true)
    print(y_predict)

    # 绘制标准化后的降雨 曲线图
    fig = plt.figure(figsize=(10, 5))  # 画板大小
    ax = fig.subplots(nrows=3, ncols=1, sharex=False)#子图shape
    # 一张画板多张子图
    actual1 = y_true[:,0]
    actual2 = y_true[:,1]
    actual3 = y_true[:,2]
    predict1 = y_predict[:,0]
    predict2 = y_predict[:,1]
    predict3 = y_predict[:,2]

    ax[0].plot(actual1,'r',label='actual1')
    ax[0].plot(predict1, 'g',label='predict1')

    ax[1].plot(actual2, 'r',label='actual2')
    ax[1].plot(predict2, 'g',label='predict2')

    ax[2].plot(actual3, 'r',label='actual3')
    ax[2].plot(predict3, 'g',label='predict3')

    plt.legend()  # 注释
    plt.grid()  # 网格
    plt.show()

