import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neural_network ,preprocessing
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit # 分层分割
from sklearn.metrics import roc_curve,auc,confusion_matrix
from scipy import interpolate
from openpyxl import Workbook
from collections import Counter
import time
import joblib
import math
import xlrd
import xlsxwriter
import random
import os
np.set_printoptions(threshold=np.inf)
dir = r"G:\20220728\human genes"

filename_excel = []
frames = []
for root, dirs, files in os.walk(dir):
    for file in files:
        #print(os.path.join(root,file))
        filename_excel.append(os.path.join(root,file))
        df = pd.read_excel(os.path.join(root,file))
        frames.append(df)

print(filename_excel)
result = pd.concat(frames)
result.to_excel(r'G:\20220728\human genes\human genes.xlsx',index = False)#保存合并的数据到电脑D盘的merge文件夹中，并把合并后的文件命名为a12.csv
def get_data():
    df1 = pd.read_excel(r"G:\BPNN_model\dataset\Positive data set.xlsx", sheet_name="Sheet1")
    data2_np = df1.iloc[:, 3:83].values
    label2_np = df1.iloc[:, 83].values
    df = pd.read_excel(r"G:\BPNN_model\dataset\Negative data set.xlsx", sheet_name="Sheet1")
    data1 = df.iloc[:, 2:82].values
    data1_np = data1[random.sample(range(0, len(data1)), len(data2_np))]
    label = df.iloc[:, 82].values
    label1_np = label[random.sample(range(0, len(data1)), len(data2_np))]
    return data1_np,label1_np,data2_np,label2_np
def MAXmin(train_feature,test_feature):
    min_max_scaler = preprocessing.MinMaxScaler().fit(train_feature)
    train_minmax = min_max_scaler.transform(train_feature)
    test_minmax = min_max_scaler.transform(test_feature)
    return train_minmax, test_minmax

def regression_method(model, x_train, y_train, x_test, y_test, x_pred):
    model.fit(x_train, y_train)   # Training samples using the training set
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    pred = model.predict(x_pred)
    #######Calculate model parameters#######
    ResidualSquare = (result - y_test)**2     #Calculate the squared residuals
    MSE = np.mean(ResidualSquare)       #Calculating the mean squared deviation
    RSS = sum(ResidualSquare)   #Calculate the sum of squared residuals
    print(f'R^2={score}')
    print(f'MSE={MSE}')
    print(f'RSS={RSS}')
    return pred

def write_predict(x_pred, pred, proba, OutPut):
    workbook = xlsxwriter.Workbook(str(OutPut))   # Save Address
    worksheet = workbook.add_worksheet('Sheet1')
    for i in range(len(x_pred)):
        for j in range(len(x_pred[0])):
            worksheet.write(i , j, x_pred[i][j])  # Write in prediction sample
        worksheet.write(i, j+1, pred[i])  # Write in predicted values
        worksheet.write(i, j+2, str(proba[i]))  # Write in predicted values
    workbook.close()
    print('数据写入完成')


model_MLP3 = neural_network.MLPClassifier(solver='sgd', activation='logistic', alpha=0.0001, hidden_layer_sizes=(37),
                            max_iter=5000, verbose=10,learning_rate="adaptive",
                           learning_rate_init=0.098,momentum=0.9)
model_MLP2 = neural_network.MLPClassifier(solver='sgd', activation='tanh', alpha=0.0001, hidden_layer_sizes=(20),
                           max_iter=5000, verbose=10,learning_rate="adaptive",
                        learning_rate_init=0.048,momentum=0.9)
model_MLP1 = neural_network.MLPClassifier(solver='sgd', activation='relu', alpha=0.0001, hidden_layer_sizes=(21),
                                   max_iter=5000, verbose=10, learning_rate="adaptive",
                                   learning_rate_init=0.048, momentum=0.9)


if __name__ == '__main__':
    neg_data, neg_label, pos_data, pos_label = get_data()
    data = np.vstack((neg_data, pos_data))
    label = np.hstack((neg_label, pos_label))
    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=48)  # 随机划分法,划分中每个类的比例和完整数据集中的相同
    for train_index, test_index in skf.split(data, label):
        train_feature = np.array(data)[train_index]
        train_label = np.array(label)[train_index]
        test_feature = np.array(data)[test_index]
        test_label = np.array(label)[test_index]
        train_minmax, test_minmax = MAXmin(train_feature, test_feature)
        ##Level 1###
        result1 = []
        NP1 = []
        remove_result1 = []
        with open('neural_network1_output.txt', 'a', encoding='utf-8') as f:
            for i in range(50):
                df21 = pd.read_excel(r"G:\BPNN_model\dataset\human genes.xlsx", sheet_name="Sheet1")
                new_sample = df21.iloc[:, 5:85].values
                article_gene_name = df21.iloc[:, 0].values
                NP_ID = df21.iloc[:, 3].values
                min_max_scaler = preprocessing.MinMaxScaler().fit(train_feature)
                predict_sample = min_max_scaler.transform(new_sample)
                method = model_MLP1
                pred = regression_method(method, train_minmax, train_label, test_minmax, test_label, predict_sample)
                y_pred1 = method.predict(predict_sample)
                pro = method.predict_proba(predict_sample)
                ZV1 = np.where(y_pred1 == 1)
                # print(ZV[0])
                for n in ZV1[0]:
                    name1 = article_gene_name[n]
                    result1.append(name1)
                    np_id = NP_ID[n]
                    NP1.append(np_id)
                for char in set(result1):
                    remove_result1.append(char)
                f.write(str(Counter(remove_result1)) + "/n")
                # output = r'G:\BPNN_model\1027.xlsx'
                # write_predict(predict_sample, pred, pro, output)
        ##Level 2###
        result2 = []
        NP2 = []
        remove_result2 = []
        with open('neural_network2_output.txt', 'a', encoding='utf-8') as f:
            for i in range(50):
                df22 = pd.read_excel(r"G:\BPNN_model\dataset\BPNN level1 output.xlsx", sheet_name="Sheet1")
                new_sample2 = df22.iloc[:, 5:85].values
                article_gene_name2 = df22.iloc[:, 0].values
                NP_ID2 = df22.iloc[:, 3].values
                min_max_scaler = preprocessing.MinMaxScaler().fit(train_feature)
                predict_sample2 = min_max_scaler.transform(new_sample2)
                method2 = model_MLP2
                pred = regression_method(method2, train_minmax, train_label, test_minmax, test_label, predict_sample2)
                y_pred2 = method2.predict(predict_sample2)
                pro2 = method2.predict_proba(predict_sample2)
                ZV2 = np.where(y_pred2 == 1)
                # print(ZV[0])
                for n in ZV2[0]:
                    name2 = article_gene_name[n]
                    result2.append(name2)
                    np_id2 = NP_ID[n]
                    NP2.append(np_id2)
                for char in set(result2):
                    remove_result2.append(char)
                f.write(str(Counter(remove_result2)) + "/n")

                # output = r'G:\BPNN_model\1027.xlsx'
                # write_predict(predict_sample, pred, pro, output)
        ##Level 3##
        result3 = []
        NP3 = []
        remove_result3 = []
        with open('neural_network3_output.txt', 'a', encoding='utf-8') as f:
            for i in range(50):
                df23 = pd.read_excel(r"G:\BPNN_model\dataset\BPNN level2 output.xlsx", sheet_name="Sheet1")
                new_sample3 = df23.iloc[:, 5:85].values
                article_gene_name3 = df23.iloc[:, 0].values
                NP_ID3 = df23.iloc[:, 1].values
                min_max_scaler = preprocessing.MinMaxScaler().fit(train_feature)
                predict_sample3 = min_max_scaler.transform(new_sample3)
                method3 = model_MLP3
                pred3 = regression_method(method3, train_minmax, train_label, test_minmax, test_label, predict_sample3)
                y_pred3 = method3.predict(predict_sample3)
                pro3 = method3.predict_proba(predict_sample3)
                ZV3 = np.where(y_pred3 == 1)
                # print(ZV[0])
                for n in ZV3[0]:
                    name3 = article_gene_name3[n]
                    result3.append(name3)
                    np_id3 = NP_ID3[n]
                    NP3.append(np_id3)
                for char in set(result3):
                    remove_result3.append(char)
                f.write(str(Counter(remove_result3)) + "/n")
                # output = r'G:\BPNN_model\1027.xlsx'
                # write_predict(predict_sample, pred, pro, output)
