# -*- coding: utf-8 -*-
from numpy import *
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import xlrd
import matplotlib.pyplot as plt

def rgb2gray(R, G, B):#定义一个rgb转灰度值的函数
    return  3 * R + 6 * G + 1 * B

def ture2value(image_ture):#定义一个envi真值图转成类别的函数，但注意：第三个for循环的颜色及类别需要根据具体的图像做出调整(自定义)
    img = array(Image.open(image_ture))#另外注意函数输入的参数为一幅图像格式为'xxx.png'
    img_true = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img_true.append(img[i, j])#将每个像元的RGB按照三维维图像的行列数读到二维数组里的RGB
    img_label = []
    for i in range(img_true.__len__()):
        img_label.append(rgb2gray(img_true[i][0], img_true[i][1], img_true[i][2]))#二维rbg专成一维灰度值数组

    for i in range(img_true.__len__()):#根据灰度值判断类别，这里可以自定义，这里是五类，如需要也可以再添加
        if img_label[i] == 1785:  # 湖泊
            img_label[i] = 4
        elif img_label[i] == 2295:
            img_label[i] = 5
        elif img_label[i] == 765:  # 绿地
            img_label[i] = 3
        elif img_label[i] == 1530:
            img_label[i] = 2
        else:
            img_label[i] = 1
    return img_label#这里返回的img_label是一个关于类别的一维数组，输出的时候需要reshape成需要的矩阵形状

def random_forest_color(train, predict):#注意这里输入的训练集为train为被改造好的xls的文件，参数格式为'xxx.xls'；另外predict为被预测图片，参数格式为'xxx.png'
    train_set = xlrd.open_workbook(train)  # 从制作好的excel表格里导入训练数据（训练集）
    table = train_set.sheets()[0]  # 从excel表第一个sheet导入
    nrows = table.nrows  # 读取训练集行数nrows
    ncols = table.ncols  # 读取训练集列数ncols

    X = []  # 读取到的234列分别代表RGB值，读取并生成二维训练集数组；(制作训练集二维数组X)
    for i in range(0, nrows):
        R = table.cell(i, 1).value
        G = table.cell(i, 2).value
        B = table.cell(i, 3).value
        X.append([R, G, B])
    X = array(X, 'f')#这里f代表float

    y = []  # 读取到的第1列代表训练集label，该数组元素个数与X的维数相同；(制作训练集label（类别）一维数组y，其维数与X相匹配)
    for i in range(0, nrows):
        label = table.cell(i, 0).value
        y.append(label)
    X = array(X, 'f')

    img = array(Image.open(predict))  # 读取遥感图像，并生成三维数组[[[R,G,B]]]
    x_test = []  # 将读取的图像生成的三维数组改化成二维测试集数组[[R,G,B]],即一排[R,G,B]；（制作被测试集的数据x_test二维数组，格式与X相同，即一排rgb）
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            x_test.append(img[i, j])
    x_test = array(x_test, 'f')

    # ------------------定义随机森林-------------------


    rf = RandomForestClassifier()  # 定义一个随机森林类
    rf.fit(X, y)  # 导入训练集X，及其对应label
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                           oob_score=False, random_state=None, verbose=0,
                           warm_start=False)  # 设置随机森林参数
    label_set = rf.predict(x_test)  # 根据导入训练集对测试集进行预测
    image_result = label_set.reshape(img.shape[0], img.shape[1])  # 生成的label为原图像大小的数组即一维，需将一维数组改化成二维数组以成图
    # print image_result
    # plt.imshow(image_result)
    # plt.show()
    return image_result#生成关于类别的二维数组

# ---------------------精度评定-------------------------
def my_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print "confusion_matrix(left labels: y_true, up labels: y_pred):"
    print "labels\t",
    for i in range(len(labels)):
        print labels[i], "\t",
    print
    for i in range(len(conf_mat)):
        print i, "\t",
        for j in range(len(conf_mat[i])):
            print conf_mat[i][j], '\t',
        print
    print

def my_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    print "classification_report(left: labels):"
    print classification_report(y_true, y_pred)

