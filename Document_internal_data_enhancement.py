# coding=utf-8
import os.path
from __init__ import SS3
from util import Dataset
import random as ra
from sklearn.metrics import f1_score,confusion_matrix
from sklearn import metrics
import math
import GF_SCORE
import pickle

class DAEB:
    clf = SS3()
    GF = []
    prediction = []
    __strength__ = []

    def gmean(matrixes):
        '''gmean函数'''
        m = matrixes
        Recall = m[1][1] / (m[1][1] + m[1][0])
        Specificity = m[0][0] / (m[0][1] + m[0][0])
        GMean_Score = math.sqrt(Recall * Specificity)
        # print("G-Mean Score: "+str(GMean_Score))
        return GMean_Score

    def build_data(x, y, n, strength: int):
        '''
        :param x: x_data
        :param y: y_data
        :param n: 划分比例
        :param strength: 文档内增强系数
        :return: x_train,y_train,x_test,y_test
        '''

        len_data = len(y)
        len_po = 0
        len_ne = 0
        for i in range(0, int(len_data)):
            if y[i] == 'po':
                len_po = len_po + 1
            if y[i] == 'ne':
                len_ne = len_ne + 1

        x_test = []
        y_test = []
        x_train = []
        y_train = []
        '''ne'''
        arr_ne = ra.sample(range(0, len_ne), len_ne)
        for i in range(0, int(len_ne / n)):
            x_test.append(x[arr_ne[i]])
            y_test.append(y[arr_ne[i]])

        for i in range(0, int((n - 1) * len_ne / n)):
            a = x[arr_ne[int(len_ne / n) + i]]
            x_train.append(strength * a[500:-100] + x[arr_ne[int(len_ne / n) + i]])
            y_train.append(y[arr_ne[int(len_ne / n) + i]])

        '''po'''
        arr_po = ra.sample(range(len_ne, len_ne + len_po), len_po)
        for j in range(0, int(len_po / n)):
            x_test.append(x[arr_po[j]])
            y_test.append(y[arr_po[j]])

        for j in range(0, int((int((n - 1) * len_po / n)) / 1)):
            x_train.append(x[arr_po[int(len_po / n) + j]])
            y_train.append(y[arr_po[int(len_po / n) + j]])

        return x_train, y_train, x_test, y_test

    def random_sample(X_train, Y_train, rate):
        """
        分层随机抽样:(保证少量数据集先进行数据载入）
        :param X_train: 待采样样本
        :param Y_train: 采样个数
        :param rate:采样比例
        """
        x_train = []
        y_train = []
        '''计算采样规模'''
        lenth = len(Y_train)
        lenth_ne = 0
        lenth_po = 0
        for i in range(0, lenth):
            if Y_train[i] == 'ne':
                lenth_ne = lenth_ne + 1
            else:
                lenth_po = lenth_po + 1
        '''ne层采样'''
        for i in range(0, int(lenth_ne * rate)):  # 设置采样次数
            random = ra.randint(1, lenth_ne - 1)
            x_train.append(X_train[random])
            y_train.append(Y_train[random])

        '''po层采样'''
        for j in range(0, int(lenth_po * rate)):
            random2 = ra.randint(lenth_ne, lenth_po - 1)
            x_train.append(X_train[random2])
            y_train.append(Y_train[random2])

        return x_train, y_train

    def bagging_train(X_train, Y_train, x_test, y_test, rate, threshold):
        '''
        :param X_train: 训练集
        :param Y_train: 标签
        :param x_test: 测试集
        :param y_test: 标签
        :param rate: 采样频率
        :param threshold: 丢弃阈值
        :return
        '''
        '''训练第i个模型'''
        Invalid_model = 0
        for i in range(0, 5):
            '''5个模型集成'''
            x_train, y_train = DAEB.random_sample(X_train, Y_train, rate)
            DAEB.clf.set_hyperparameters(s=0.32, l=1.62, p=1.1)
            DAEB.clf.train(x_train, y_train, n_grams=i + 2)
            y_pred = DAEB.clf.predict(1, x_test)
            '''保存预测结果'''
            gf_score = GF_SCORE.gf_score(y_test, y_pred, 1, 1)
            if gf_score > threshold:
                if i == 0:
                    pickle.dump(DAEB.clf, file=open('./model1.pkl', 'wb+'))
                if i == 1:
                    pickle.dump(DAEB.clf, file=open('./model2.pkl', 'wb+'))
                if i == 2:
                    pickle.dump(DAEB.clf, file=open('./model3.pkl', 'wb+'))
                if i == 3:
                    pickle.dump(DAEB.clf, file=open('./model4.pkl', 'wb+'))
                if i == 4:
                    pickle.dump(DAEB.clf, file=open('./model5.pkl', 'wb+'))
            else:
                Invalid_model = Invalid_model + 1
        if Invalid_model == 5:
            quit('丢弃阈值过高，无可用模型，请重新设置！')

        print('训练完成')

    def bagging_predict(x_test):
        '''训练第i个模型'''
        y_pred = []
        live_learner = 0
        for i in range(0, 5):
            '''5个模型集成'''
            if i == 0 and os.path.isfile('model1.pkl'):
                DAEB.clf = pickle.load(file=open('./model1.pkl', 'rb'))
                os.remove('./model1.pkl')
                live_learner = live_learner + 1
            if i == 1 and os.path.isfile('model2.pkl'):
                DAEB.clf = pickle.load(file=open('./model2.pkl', 'rb'))
                os.remove('./model2.pkl')
                live_learner = live_learner + 1
            if i == 2 and os.path.isfile('model3.pkl'):
                DAEB.clf = pickle.load(file=open('./model3.pkl', 'rb'))
                os.remove('./model3.pkl')
                live_learner = live_learner + 1
            if i == 3 and os.path.isfile('model4.pkl'):
                DAEB.clf = pickle.load(file=open('./model4.pkl', 'rb'))
                os.remove('./model4.pkl')
                live_learner = live_learner + 1
            if i == 4 and os.path.isfile('model5.pkl'):
                DAEB.clf = pickle.load(file=open('./model5.pkl', 'rb'))
                os.remove('./model5.pkl')
                live_learner = live_learner + 1

            DAEB.clf.set_hyperparameters(s=0.32, l=1.62, p=1.1)
            y_pred.append(DAEB.clf.predict(1, x_test))

        '''投票得出结果'''
        if live_learner == 1:
            return y_pred[0]

        Y_pred = []
        test_len = len(y_pred[0])
        if live_learner > 1:
            for i in range(0, test_len):
                live_flag = live_learner
                votes_number = 1
                if live_flag > 1:
                    if y_pred[0][i] == y_pred[1][i]:
                        votes_number = votes_number + 1
                        live_flag = live_flag - 1
                    else:
                        different = y_pred[1][i]
                        live_flag = live_flag - 1
                if live_flag > 1:
                    if y_pred[0][i] == y_pred[2][i]:
                        votes_number = votes_number + 1
                        live_flag = live_flag - 1
                    else:
                        different = y_pred[2][i]
                        live_flag = live_flag - 1
                if live_flag > 1:
                    if y_pred[0][i] == y_pred[3][i]:
                        votes_number = votes_number + 1
                        live_flag = live_flag - 1
                    else:
                        different = y_pred[3][i]
                        live_flag = live_flag - 1
                if live_flag > 1:
                    if y_pred[0][i] == y_pred[4][i]:
                        votes_number = votes_number + 1
                    else:
                        different = y_pred[4][i]
                if votes_number > (live_learner / 2):  # 投票数大于存活学习器的一半
                    Y_pred.append(y_pred[0][i])
                else:
                    Y_pred.append(different)

        return Y_pred

    def predict(x_test):
        '''训练第i个模型'''
        y_pred = []
        live_learner = 0
        for i in range(0, 5):
            '''5个模型集成'''
            if i == 0 and os.path.isfile('model1.pkl'):
                DAEB.clf = pickle.load(file=open('./model1.pkl', 'rb'))
                live_learner = live_learner + 1
            if i == 1 and os.path.isfile('model2.pkl'):
                DAEB.clf = pickle.load(file=open('./model2.pkl', 'rb'))
                live_learner = live_learner + 1
            if i == 2 and os.path.isfile('model3.pkl'):
                DAEB.clf = pickle.load(file=open('./model3.pkl', 'rb'))
                live_learner = live_learner + 1
            if i == 3 and os.path.isfile('model4.pkl'):
                DAEB.clf = pickle.load(file=open('./model4.pkl', 'rb'))
                live_learner = live_learner + 1
            if i == 4 and os.path.isfile('model5.pkl'):
                DAEB.clf = pickle.load(file=open('./model5.pkl', 'rb'))
                live_learner = live_learner + 1

            DAEB.clf.set_hyperparameters(s=0.32, l=1.62, p=1.1)
            y_pred.append(DAEB.clf.predict(1, x_test))

        '''投票得出结果'''
        if live_learner == 1:
            return y_pred[0]

        Y_pred = []
        test_len = len(y_pred[0])
        if live_learner > 1:
            for i in range(0, test_len):
                live_flag = live_learner
                votes_number = 1
                if live_flag > 1:
                    if y_pred[0][i] == y_pred[1][i]:
                        votes_number = votes_number + 1
                        live_flag = live_flag - 1
                    else:
                        different = y_pred[1][i]
                        live_flag = live_flag - 1
                if live_flag > 1:
                    if y_pred[0][i] == y_pred[2][i]:
                        votes_number = votes_number + 1
                        live_flag = live_flag - 1
                    else:
                        different = y_pred[2][i]
                        live_flag = live_flag - 1
                if live_flag > 1:
                    if y_pred[0][i] == y_pred[3][i]:
                        votes_number = votes_number + 1
                        live_flag = live_flag - 1
                    else:
                        different = y_pred[3][i]
                        live_flag = live_flag - 1
                if live_flag > 1:
                    if y_pred[0][i] == y_pred[4][i]:
                        votes_number = votes_number + 1
                    else:
                        different = y_pred[4][i]
                if votes_number > (live_learner / 2):  # 投票数大于存活学习器的一半
                    Y_pred.append(y_pred[0][i])
                else:
                    Y_pred.append(different)

        return Y_pred

    def DAEB_bagging_fit(X_DATA, Y_DATA, rate, down: int, up: int, threshold):
        """
        分层增强随机抽样:(保证少量数据集先进行数据载入）
            :param X_DATA,Y_DATA: 待采样样本
            :param rate: 采样频率
            down:搜索下界
            up:搜索上界
            return 最优预测结果
        """
        for i in range(down, up):
            X_train, Y_train, x_test, y_test = DAEB.build_data(X_DATA, Y_DATA, 5, i)
            DAEB.bagging_train(X_train, Y_train, x_test, y_test, rate, threshold)
            y_pred = DAEB.bagging_predict(x_test)
            # GF
            GFScore = GF_SCORE.gf_score(y_test, y_pred, 1, 1)
            DAEB.GF.append(GFScore)
            DAEB.prediction.append(y_pred)
            DAEB.__strength__.append(i)
        index = DAEB.GF.index(max(DAEB.GF))
        best = DAEB.__strength__[index]
        pre = DAEB.prediction[index]
        print("最优采样系数: ", best)
        X_train, Y_train, x_test, y_test = DAEB.build_data(X_DATA, Y_DATA, 5, best)
        DAEB.bagging_train(X_train, Y_train, x_test, y_test, rate, threshold)
        return pre

#X_DATA , Y_DATA = Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data)/kflod")
#pre,y_test=DAEB_bagging_fit(X_DATA,Y_DATA,2/3,1,3,0.6)






