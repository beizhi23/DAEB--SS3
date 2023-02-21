import util
from __init__ import SS3
from util import Dataset, Evaluation, span
#from server import Live_Test
from random import randint as ra
from sklearn.metrics import accuracy_score
from sklearn import metrics
import math
import Gmean_search


util.Evaluation.clear_cache()#先清除内存
clf = SS3()

s, l, p, _ = clf.get_hyperparameters()

print("Smoothness(s):", s)
print("Significance(l):", l)
print("Sanction(p):", p)

#x_train, y_train = Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data) - 副本 - 副本/train")
#x_test, y_test = Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data) - 副本 - 副本/test")

'''构造测试集和训练集'''
x , y = Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data)/kflod")
def builddata(x,y,n):
    len_data = len(y)
    len_po = 0
    len_ne = 0
    for i in range(0, int(len_data)):
        if y[i] == 'po':
            len_po = len_po+1
        if y[i] == 'ne':
            len_ne = len_ne+1

    x_test=[]
    y_test=[]
    x_train=[]
    y_train=[]

    '''ne'''
    for i in range(0, int(len_ne/n)):
        random = ra(1, len_ne)
        x_test.append(x[random])
        y_test.append(y[random])

    for i in range(0, int((n-1)*len_ne/n)):
        random = ra(1, len_ne)
        x_train.append(x[random])
        y_train.append(y[random])

    '''po'''
    for j in range(0, int(len_po/n)):
        random = ra(len_ne, len_ne+len_po)
        x_test.append(x[random])
        y_test.append(y[random])

    for j in range(0, int((n-1)*len_po/n)):
        random = ra(len_ne, len_ne+len_po-1)
        x_train.append(x[random])
        y_train.append(y[random])

    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test=builddata(x,y,4)



clf.train(x_train, y_train, n_grams=2)
clf.set_hyperparameters(s=0.32, l=2.0, p=0.8)
y_pred = clf.predict(x_test)

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

clf.save_wordcloud("po",plot=True)  #正类
clf.save_wordcloud("ne", color="tomato", plot=True) #负类
clf.save_wordcloud("po", n_grams=3, plot=True) #3-gram 正类
clf.save_wordcloud("ne", n_grams=3, color="tomato", plot=True) #3-gram 负类

Evaluation.test(clf, x_test, y_test)#绘制混淆矩阵


"""网格调参"""
'''
s_vals=span(0.2, 0.8, 6)  # [0.2 , 0.32, 0.44, 0.56, 0.68, 0.8]
l_vals=span(0.1, 2, 6)    # [0.1 , 0.48, 0.86, 1.24, 1.62, 2]
p_vals=span(0.5, 2, 6)    # [0.5, 0.8, 1.1, 1.4, 1.7, 2]
s,l,p=Gmean_search.GMean_Search(clf,x_test,y_test,s_vals,l_vals,p_vals)
print("best_s:",s)
print("best_l:",l)
print("best_p:",p)
'''


clf = SS3(name="despression")

clf.train(x_train, y_train)
s_vals=span(0.2, 0.8, 6)  # [0.2 , 0.32, 0.44, 0.56, 0.68, 0.8]
l_vals=span(0.1, 2, 6)    # [0.1 , 0.48, 0.86, 1.24, 1.62, 2]
p_vals=span(0.5, 2, 6)    # [0.5, 0.8, 1.1, 1.4, 1.7, 2]

best_s, best_l, best_p, _ = Evaluation.grid_search(
    clf, x_test, y_test,
    s=s_vals, l=l_vals, p=p_vals,
    tag="grid search (test)"  # <-- this is optional! >_<
)

print("The hyperparameter values that obtained the best Accuracy are:")
print("Smoothness(s):", best_s)
print("Significance(l):", best_l)
print("Sanction(p):", best_p)



#提高f1分数的超参数值
s, l, p, _ = Evaluation.get_best_hyperparameters(metric="f1-score", metric_target="macro avg")
print("s=%.2f, l=%.2f, and p=%.2f" % (s, l, p))

#提高recall分数的超参数值
s, l, p, _ = Evaluation.get_best_hyperparameters(metric="recall", metric_target="macro avg")
print("s=%.2f, l=%.2f, and p=%.2f" % (s, l, p))

'''
import pickle
model_SS3='ufo-model.pkl'
file=open(model_SS3,'wb')
pickle.dump(clf,file)
file.close()

file=open('ufo-model.pkl','rb')
model = pickle.load(file)
file.close()
print(model.predict(x_test))
'''
