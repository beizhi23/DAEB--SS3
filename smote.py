from sklearn.preprocessing import LabelBinarizer
from __init__ import SS3
from util import Dataset, Evaluation, span
#from server import Live_Test
from sklearn.metrics import accuracy_score
from sklearn import metrics
import math


x_train,y_train = Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data) - 副本 - 副本/train")

'''smote采样'''

from imblearn.over_sampling import SMOTE

sm = SMOTE()

lb=LabelBinarizer()

x_train_bin = lb.fit_transform(x_train)
x_train_res, y_train_res = sm.fit_resample(x_train_bin, y_train)
x_train=lb.inverse_transform(x_train_res)
y_train=y_train_res

'''使用smote采样后数据训练'''

x_test,y_test = Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data) - 副本 - 副本/test")

clf=SS3()
clf.train(x_train, y_train, n_grams=3)
clf.set_hyperparameters(s=0.44, l=2.0, p=0.5)
y_pred = clf.predict(x_test)
print(y_pred)

print("Accuracy:", accuracy_score(y_pred, y_test))

print("f1_score",metrics.f1_score(y_test,y_pred,labels=None,pos_label='po'))
print("recall",metrics.recall_score(y_test,y_pred,labels=None,pos_label='po'))

matrixes=metrics.confusion_matrix(y_test,y_pred,labels=['ne','po'])
print(matrixes)

'''计算G-Mean评分'''
m=matrixes
Recall=m[1][1]/(m[1][1]+m[1][0])
Specificity=m[0][0]/(m[0][1]+m[0][0])
GMean_Score=math.sqrt(Recall*Specificity)
print("G-Mean Score: "+str(GMean_Score))



