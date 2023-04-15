from sklearn.metrics import f1_score,confusion_matrix
from Gmean_search import gmean


def gf_score(y_true,y_test,a,b):
    '''
    不平衡数据上平衡GMEAN SCORE和F1 SCORE的评价指标
    :param y_true: 真实标签
    :param y_test: 测试标签
    :param a: g_mean权重
    :param b: f1权重
    :return: gf_score
    '''
    f1=f1_score(y_true,y_test,pos_label='po')
    m=confusion_matrix(y_true,y_test,labels=['ne','po'])
    g_mean=gmean(m)
    gf_score=(2*g_mean*f1)/(b*g_mean+a*f1)
    print('F1-Score:',f1)
    print('GF-Score:',gf_score)
    return gf_score