import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os

# The first kind of confusion matrix
"""def plot_confusion_matrix_1(pre_lab, Y_test, num_p, logger):
    acc = accuracy_score(Y_test, pre_lab)
    logger.info('The prediction accuracy on this fold is: {:.4f}\n'.format(acc))

    logger.info('Ground of Truth')#给出数据的相关信息概览：行数、列数、列索引等
    logger.info(Y_test)
    logger.info('Predicted label')
    logger.info(pre_lab)


    class_label = ['ANGER', 'CONTEMPT','DISGUST', 'FEAR', 'HAPPY','NEUTRAL', 'SADNESS', 'SURPRISE']
    conf_mat = confusion_matrix(Y_test, pre_lab)
    df_cm = pd.DataFrame(
        conf_mat,
        index=class_label,
        columns=class_label
    )

    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=25, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Backbone_d--{} time 10-fold-cross-validation cm_1.png'.format(num_p))
    plt.show()

    return
    
def get_UAR(trues_te,pres_te):
    #混淆矩阵计算
    cm=confusion_matrix(trues_te,pres_te)
    # 获得样本值、预测值类别 样本数;每个样本的召回率recall =每个类别正确预测样本数/每个类别真实样本数
    acc_per_cls=[cm[i,i]/sum(cm[i]) for i in range(len(cm))]
    # 计算UAR的值，通过召回率计算
    UAR=sum(acc_per_cls)/len(acc_per_cls)
    return UAR"""
# The second kind of confusion matrix
def plot_confusion_matrix_2(pre_lab, Y_test):


    labels_name = ['Happiness','Sadness','Neutral','Anger','Surprise', 'Disgust','Fear' ]
    #  get_UAR
    cm = confusion_matrix(Y_test, pre_lab)
     #获得样本值、预测值类别 样本数;每个样本的召回率recall =每个类别正确预测样本数/每个类别真实样本数
    acc_per_cls = [cm[i, i] / sum(cm[i]) for i in range(len(cm))]
    # 计算UAR的值，通过召回率计算
    uar = sum(acc_per_cls) / len(acc_per_cls)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    ind_array = np.arange(len(labels_name))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if x_val == y_val:
            plt.text(x_val, y_val, "%.2f" % (c,), color='black', fontsize=10, va='center', ha='center')
        else:
            plt.text(x_val, y_val, "%.2f" % (c,), color='black', fontsize=10, va='center', ha='center')
    confusion_matrix_2(cm_normalized, labels_name, 'Normalized Confusion Matrix')
    plt.savefig("./picture/dfew.png")
    plt.close()
    # plt.show()

    return  uar



# 混淆矩阵百分比可视化
# cm为混淆矩阵，labels表示 标签名称 happeness等七个不同的标签
def confusion_matrix_2(cm, labels_name, title):
    """cm.sum(axis=1)计算混淆矩阵每个实际类别的元素和，得到以实际类别为行的向量;
    [:,np.newaxis]为保持维度一致性，在该向量上增加一个新维度，将其转换为列向量
    /并把混淆矩阵中的每个元素除以对应行的总和；*100 小数转换为百分数
    即 最终得到的混淆矩阵的每个元素表示相应类别的预测结果在总预测样本所占百分比"""
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    """imshow()可视化数据，cm被可视化数据，interpolation='nearest'最近邻插值插值方法，
    指定如何在图像中插值现实像素值，使用最近的像素值进行现实
    cmap=plt.cm.PuRd颜色映射，用于给不同的像素值分配颜色映射，使用红色和紫色渐变色
    总之：可视化混淆矩阵，图像中的每个像素对应混淆矩阵的一个元素，颜色深浅表示相应像素数值大小"""
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title(title)
    """colorbar()用于在图像一侧添加颜色条，颜色条提供了图像中不同颜色对应数值参考范围参考标尺"""
    plt.colorbar()
    """num_local 创建的NUmPy数组，并将一个范围内的整数值转换为数组
    即生成表情标签长度的7维矩阵"""
    num_local = np.array(range(len(labels_name)))
    """xticks用于设置x轴刻度的位置和标签，即将x轴刻度设置维num_local数组中的值，
    对应刻度标签设置维labels_name数组中的值，同时rotation=25可以设置x轴刻度标签旋转角度，
    以便更好显示标签内容"""
    plt.xticks(num_local, labels_name, rotation=25)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return














