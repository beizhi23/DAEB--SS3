
import math
from sklearn import metrics
from sklearn.metrics import f1_score

def matrixes(y_test,y_pred):
    matrixes=metrics.confusion_matrix(y_test,y_pred,labels=['ne','po'])
    return matrixes

def gmean(matrixes):
    '''gmean函数'''
    m=matrixes
    Recall = m[1][1] / (m[1][1] + m[1][0])
    Specificity = m[0][0] / (m[0][1] + m[0][0])
    GMean_Score=math.sqrt(Recall*Specificity)
    print(m)
    print("G-Mean Score: "+str(GMean_Score))
    return GMean_Score

def GMean_Search(clf, x_test, y_test,s_vals, l_vals, p_vals,):
    "GMean评分网格调参函数"
    ite=len(s_vals)
    G_Mean=0
    flag_s=0
    flag_l=0
    flag_p=0
    for i in range(0,ite):
        for j in range(0,ite):
            for k in range(0,ite):
                clf.set_hyperparameters(s=s_vals[i], l=l_vals[j], p=p_vals[k])
                y_pred=clf.predict(1,x_test)
                matr=matrixes(y_test,y_pred)
                g_mean=gmean(matr)
                if G_Mean<g_mean:#选择最优G_Mean评分
                    G_Mean=g_mean
                    flag_s=i
                    flag_l=j
                    flag_p=k
    s=s_vals[flag_s]
    l=l_vals[flag_l]
    p=p_vals[flag_p]
    return s,l,p
def F1_Search(clf, x_test, y_test,s_vals, l_vals, p_vals):
    "GMean评分网格调参函数"
    ite=len(s_vals)
    F1=0
    flag_s=0
    flag_l=0
    flag_p=0
    for i in range(0,ite):
        for j in range(0,ite):
            for k in range(0,ite):
                clf.set_hyperparameters(s=s_vals[i], l=l_vals[j], p=p_vals[k])
                y_pred=clf.predict(1,x_test)
                f1=f1_score(y_test,y_pred,pos_label='po')
                if F1<f1:#选择最优G_Mean评分
                    F1=f1
                    flag_s=i
                    flag_l=j
                    flag_p=k
    s=s_vals[flag_s]
    l=l_vals[flag_l]
    p=p_vals[flag_p]
    return s,l,p
def Integrated_parameter_adjustment(clf, x_test, y_test,s_vals, l_vals, p_vals,f1_down):
    "GMean and f1 评分网格调参函数"
    ite=len(s_vals)
    G_Mean=0
    flag_s=0
    flag_l=0
    flag_p=0
    for i in range(0,ite):
        for j in range(0,ite):
            for k in range(0,ite):
                clf.set_hyperparameters(s=s_vals[i], l=l_vals[j], p=p_vals[k])
                y_pred=clf.predict(1,x_test)
                matr=matrixes(y_test,y_pred)
                g_mean=gmean(matr)
                f1=f1_score(y_test, y_pred, pos_label='po')
                print("f1 score: ",f1)
                if G_Mean<g_mean and f1>f1_down:#在f1范围内选择最优G_Mean评分
                    G_Mean=g_mean
                    flag_s=i
                    flag_l=j
                    flag_p=k
    s=s_vals[flag_s]
    l=l_vals[flag_l]
    p=p_vals[flag_p]
    return s,l,p

def GF_search(clf, x_test, y_test,s_vals, l_vals, p_vals):
    "GF评分网格调参函数"
    ite=len(s_vals)
    GF=0
    flag_s=0
    flag_l=0
    flag_p=0
    for i in range(0,ite):
        for j in range(0,ite):
            for k in range(0,ite):
                clf.set_hyperparameters(s=s_vals[i], l=l_vals[j], p=p_vals[k])
                y_pred=clf.predict(1,x_test)
                matr=matrixes(y_test,y_pred)
                g_mean=gmean(matr)
                f1=f1_score(y_test, y_pred, pos_label='po')
                print("f1 score: ",f1)
                gf=(2*g_mean*f1)/(g_mean+f1)
                if GF<gf:
                    GF=gf
                    flag_s=i
                    flag_l=j
                    flag_p=k
    s=s_vals[flag_s]
    l=l_vals[flag_l]
    p=p_vals[flag_p]
    return s,l,p




