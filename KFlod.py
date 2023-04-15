import random as ra

def kflod(x,y,n):
    '''
    :param x: x_data
    :param y: y_data
    :param n: 划分比例
    :return: x_train,y_train
    '''

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
    arr_ne=ra.sample(range(0,len_ne),len_ne)
    for i in range(0, int(len_ne/n)):
        x_test.append(x[arr_ne[i]])
        y_test.append(y[arr_ne[i]])

    for i in range(0, int((n - 1) * len_ne / n)):
        x_train.append(x[arr_ne[int(len_ne / n) + i]])
        y_train.append(y[arr_ne[int(len_ne / n) + i]])

    '''po'''
    arr_po=ra.sample(range(len_ne, len_ne+len_po),len_po)
    for j in range(0, int(len_po/n)):
        x_test.append(x[arr_po[j]])
        y_test.append(y[arr_po[j]])

    for j in range(0, int((int((n-1)*len_po/n))/1)):
        x_train.append(x[arr_po[int(len_po/n)+j]])
        y_train.append(y[arr_po[int(len_po/n)+j]])

    return x_train,y_train,x_test,y_test

def stacking_flod(x,y):
    len_data = len(y)
    len_po = 0
    len_ne = 0
    for i in range(0, int(len_data)):
        if y[i] == 'po':
            len_po = len_po + 1
        if y[i] == 'ne':
            len_ne = len_ne + 1
    x_train1=[]
    y_train1=[]
    x_train2=[]
    y_train2=[]
    x_train3=[]
    y_train3=[]
    x_train4=[]
    y_train4=[]
    x_train5=[]
    y_train5=[]
    x_test1=[]
    y_test1=[]
    x_test2=[]
    y_test2=[]
    x_test3=[]
    y_test3=[]
    x_test4=[]
    y_test4=[]
    x_test5=[]
    y_test5=[]
    '''ne'''
    arr_ne = ra.sample(range(0, len_ne), len_ne)
    flod_ne=int(len_ne/5)
    for i in range(0,flod_ne):
        x_test1.append(x[arr_ne[i]])
        y_test1.append(y[arr_ne[i]])
    for i in range(flod_ne,2*flod_ne):
        x_test2.append(x[arr_ne[i]])
        y_test2.append(y[arr_ne[i]])
    for i in range(2*flod_ne,3*flod_ne):
        x_test3.append(x[arr_ne[i]])
        y_test3.append(y[arr_ne[i]])
    for i in range(3*flod_ne,4*flod_ne):
        x_test4.append(x[arr_ne[i]])
        y_test4.append(y[arr_ne[i]])
    for i in range(4*flod_ne,5*flod_ne):
        x_test5.append(x[arr_ne[i]])
        y_test5.append(y[arr_ne[i]])

    for i in range(flod_ne,5*flod_ne):
        x_train1.append(x[arr_ne[i]])
        y_train1.append(y[arr_ne[i]])
    for i in range(2*flod_ne,5*flod_ne):
        x_train2.append(x[arr_ne[i]])
        y_train2.append(y[arr_ne[i]])
    for i in range(0,flod_ne):
        x_train2.append(x[arr_ne[i]])
        y_train2.append(y[arr_ne[i]])
    for i in range(3*flod_ne,5*flod_ne):
        x_train3.append(x[arr_ne[i]])
        y_train3.append(y[arr_ne[i]])
    for i in range(0,2*flod_ne):
        x_train3.append(x[arr_ne[i]])
        y_train3.append(y[arr_ne[i]])
    for i in range(4*flod_ne,5*flod_ne):
        x_train4.append(x[arr_ne[i]])
        y_train4.append(y[arr_ne[i]])
    for i in range(0,3*flod_ne):
        x_train4.append(x[arr_ne[i]])
        y_train4.append(y[arr_ne[i]])
    for i in range(0,4*flod_ne):
        x_train5.append(x[arr_ne[i]])
        y_train5.append(y[arr_ne[i]])

    arr_po = ra.sample(range(len_ne,len_data),len_po)
    flod_po = int(len_po/5)
    for i in range(0,flod_po):
        x_test1.append(x[arr_po[i]])
        y_test1.append(y[arr_po[i]])
    for i in range(flod_po,2 * flod_po):
        x_test2.append(x[arr_po[i]])
        y_test2.append(y[arr_po[i]])
    for i in range(2 * flod_po,3 * flod_po):
        x_test3.append(x[arr_po[i]])
        y_test3.append(y[arr_po[i]])
    for i in range(3 * flod_po, 4 * flod_po):
        x_test4.append(x[arr_po[i]])
        y_test4.append(y[arr_po[i]])
    for i in range(4 * flod_po,5*flod_po):
        x_test5.append(x[arr_po[i]])
        y_test5.append(y[arr_po[i]])

    for i in range(flod_po, 5*flod_po):
        x_train1.append(x[arr_po[i]])
        y_train1.append(y[arr_po[i]])
    for i in range(2 * flod_po, 5*flod_po):
        x_train2.append(x[arr_po[i]])
        y_train2.append(y[arr_po[i]])
    for i in range(0,flod_po):
        x_train2.append(x[arr_po[i]])
        y_train2.append(y[arr_po[i]])
    for i in range(len_ne+3 * flod_po, 5*flod_po):
        x_train3.append(x[arr_po[i]])
        y_train3.append(y[arr_po[i]])
    for i in range(0,2 * flod_po):
        x_train3.append(x[arr_po[i]])
        y_train3.append(y[arr_po[i]])
    for i in range(4 * flod_po, 5* flod_po):
        x_train4.append(x[arr_po[i]])
        y_train4.append(y[arr_po[i]])
    for i in range(0,3 * flod_po):
        x_train4.append(x[arr_po[i]])
        y_train4.append(y[arr_po[i]])
    for i in range(0,4 * flod_po):
        x_train5.append(x[arr_po[i]])
        y_train5.append(y[arr_po[i]])

    return x_train1, y_train1, x_train2, y_train2, x_train3, y_train3, x_train4, y_train4, x_train5, y_train5, x_test1, y_test1, x_test2, y_test2, x_test3, y_test3, x_test4, y_test4, x_test5, y_test5

def Mul_kflod(X_text,Y_text,X_vision,Y_vision,X_audio,Y_audio,n):
    '''
    :param x: x_data
    :param y: y_data
    :param n: 划分比例
    :return: x_train,y_train
    '''

    len_data = len(Y_text)
    len_po = 0
    len_ne = 0
    for i in range(0, int(len_data)):
        if Y_text[i] == 'po':
            len_po = len_po+1
        if Y_text[i] == 'ne':
            len_ne = len_ne+1

    x_text_test=[]
    y_text_test=[]
    x_text_train=[]
    y_text_train=[]
    x_vision_test=[]
    y_vision_test=[]
    x_vision_train=[]
    y_vision_train=[]
    x_audio_test=[]
    y_audio_test=[]
    x_audio_train=[]
    y_audio_train=[]
    '''ne'''
    arr_ne=ra.sample(range(0,len_ne),len_ne)
    for i in range(0, int(len_ne/n)):
        x_text_test.append(X_text[arr_ne[i]])
        y_text_test.append(Y_text[arr_ne[i]])
        x_vision_test.append(X_vision[arr_ne[i]])
        y_vision_test.append(Y_vision[arr_ne[i]])
        x_audio_test.append(X_audio[arr_ne[i]])
        y_audio_test.append(Y_audio[arr_ne[i]])

    for i in range(0, int((n - 1) * len_ne / n)):
        x_text_train.append(X_text[arr_ne[int(len_ne / n) + i]])
        y_text_train.append(Y_text[arr_ne[int(len_ne / n) + i]])
        x_vision_train.append(X_vision[arr_ne[int(len_ne / n) + i]])
        y_vision_train.append(Y_vision[arr_ne[int(len_ne / n) + i]])
        x_audio_train.append(X_audio[arr_ne[int(len_ne / n) + i]])
        y_audio_train.append(Y_audio[arr_ne[int(len_ne / n) + i]])

    '''po'''
    arr_po=ra.sample(range(len_ne, len_ne+len_po),len_po)
    for j in range(0, int(len_po/n)):
        x_text_test.append(X_text[arr_po[j]])
        y_text_test.append(Y_text[arr_po[j]])
        x_vision_test.append(X_vision[arr_po[j]])
        y_vision_test.append(Y_vision[arr_po[j]])
        x_audio_test.append(X_audio[arr_po[j]])
        y_audio_test.append(Y_audio[arr_po[j]])

    for j in range(0, int((int((n-1)*len_po/n))/1)):
        x_text_train.append(X_text[arr_po[int(len_po/n)+j]])
        y_text_train.append(Y_text[arr_po[int(len_po/n)+j]])
        x_vision_train.append(X_vision[arr_po[int(len_po / n) + j]])
        y_vision_train.append(Y_vision[arr_po[int(len_po / n) + j]])
        x_audio_train.append(X_audio[arr_po[int(len_po / n) + j]])
        y_audio_train.append(Y_audio[arr_po[int(len_po / n) + j]])

    return x_text_train,y_text_train,x_text_test,y_text_test,x_vision_train,y_vision_train,x_vision_test,y_vision_test,x_audio_train,y_audio_train,x_audio_test,y_audio_test

def Mul_stacking_flod(x_text_train,y_text_train,x_vision_train,y_vision_train,x_audio_train,y_audio_train):
    len_data = len(y_text_train)
    len_po = 0
    len_ne = 0
    for i in range(0, int(len_data)):
        if y_text_train[i] == 'po':
            len_po = len_po + 1
        if y_text_train[i] == 'ne':
            len_ne = len_ne + 1
    x_train1=[]
    y_train1=[]
    x_train2 = []
    y_train2 = []
    x_train2_vision = []
    y_train2_vision = []
    x_train2_audio = []
    y_train2_audio = []
    x_train3=[]
    y_train3=[]
    x_train4=[]
    y_train4=[]
    x_train5=[]
    y_train5=[]
    x_test1=[]
    y_test1=[]
    x_test2=[]
    y_test2=[]
    x_test2_vision = []
    y_test2_vision = []
    x_test2_audio = []
    y_test2_audio = []
    x_test3=[]
    y_test3=[]
    x_test4=[]
    y_test4=[]
    x_test5=[]
    y_test5=[]
    '''ne'''
    'test'
    arr_ne = ra.sample(range(0, len_ne), len_ne)
    flod_ne=int(len_ne/5)
    for i in range(0,flod_ne):
        x_test1.append(x_text_train[arr_ne[i]])
        y_test1.append(y_text_train[arr_ne[i]])
    for i in range(flod_ne,2*flod_ne):
        x_test2.append(x_text_train[arr_ne[i]])
        y_test2.append(y_text_train[arr_ne[i]])
        x_test2_vision.append(x_vision_train[arr_ne[i]])
        y_test2_vision.append(y_vision_train[arr_ne[i]])
        x_test2_audio.append(x_audio_train[arr_ne[i]])
        y_test2_audio.append(y_audio_train[arr_ne[i]])
    for i in range(2*flod_ne,3*flod_ne):
        x_test3.append(x_text_train[arr_ne[i]])
        y_test3.append(y_text_train[arr_ne[i]])
    for i in range(3*flod_ne,4*flod_ne):
        x_test4.append(x_text_train[arr_ne[i]])
        y_test4.append(y_text_train[arr_ne[i]])
    for i in range(4*flod_ne,5*flod_ne):
        x_test5.append(x_text_train[arr_ne[i]])
        y_test5.append(y_text_train[arr_ne[i]])

    'train'
    for i in range(flod_ne,5*flod_ne):
        x_train1.append(x_text_train[arr_ne[i]])
        y_train1.append(y_text_train[arr_ne[i]])
    for i in range(2*flod_ne,5*flod_ne):
        x_train2.append(x_text_train[arr_ne[i]])
        y_train2.append(y_text_train[arr_ne[i]])
        x_train2_vision.append(x_vision_train[arr_ne[i]])
        y_train2_vision.append(y_vision_train[arr_ne[i]])
        x_train2_audio.append(x_audio_train[arr_ne[i]])
        y_train2_audio.append(y_audio_train[arr_ne[i]])
    for i in range(0,flod_ne):
        x_train2.append(x_text_train[arr_ne[i]])
        y_train2.append(y_text_train[arr_ne[i]])
        x_train2_vision.append(x_vision_train[arr_ne[i]])
        y_train2_vision.append(y_vision_train[arr_ne[i]])
        x_train2_audio.append(x_audio_train[arr_ne[i]])
        y_train2_audio.append(y_audio_train[arr_ne[i]])
    for i in range(3*flod_ne,5*flod_ne):
        x_train3.append(x_text_train[arr_ne[i]])
        y_train3.append(y_text_train[arr_ne[i]])
    for i in range(0,2*flod_ne):
        x_train3.append(x_text_train[arr_ne[i]])
        y_train3.append(y_text_train[arr_ne[i]])
    for i in range(4*flod_ne,5*flod_ne):
        x_train4.append(x_text_train[arr_ne[i]])
        y_train4.append(y_text_train[arr_ne[i]])
    for i in range(0,3*flod_ne):
        x_train4.append(x_text_train[arr_ne[i]])
        y_train4.append(y_text_train[arr_ne[i]])
    for i in range(0,4*flod_ne):
        x_train5.append(x_text_train[arr_ne[i]])
        y_train5.append(y_text_train[arr_ne[i]])

    '''po'''
    'test'
    arr_po = ra.sample(range(len_ne,len_data),len_po)
    flod_po = int(len_po/5)
    for i in range(0,flod_po):
        x_test1.append(x_text_train[arr_po[i]])
        y_test1.append(y_text_train[arr_po[i]])
    for i in range(flod_po,2 * flod_po):
        x_test2.append(x_text_train[arr_po[i]])
        y_test2.append(y_text_train[arr_po[i]])
        x_test2_vision.append(x_vision_train[arr_po[i]])
        y_test2_vision.append(y_vision_train[arr_po[i]])
        x_test2_audio.append(x_audio_train[arr_po[i]])
        y_test2_audio.append(y_audio_train[arr_po[i]])
    for i in range(2 * flod_po,3 * flod_po):
        x_test3.append(x_text_train[arr_po[i]])
        y_test3.append(y_text_train[arr_po[i]])
    for i in range(3 * flod_po, 4 * flod_po):
        x_test4.append(x_text_train[arr_po[i]])
        y_test4.append(y_text_train[arr_po[i]])
    for i in range(4 * flod_po,5*flod_po):
        x_test5.append(x_text_train[arr_po[i]])
        y_test5.append(y_text_train[arr_po[i]])

    'train'
    for i in range(flod_po, 5*flod_po):
        x_train1.append(x_text_train[arr_po[i]])
        y_train1.append(y_text_train[arr_po[i]])
    for i in range(2 * flod_po, 5*flod_po):
        x_train2.append(x_text_train[arr_po[i]])
        y_train2.append(y_text_train[arr_po[i]])
        x_train2_vision.append(x_vision_train[arr_po[i]])
        y_train2_vision.append(y_vision_train[arr_po[i]])
        x_train2_audio.append(x_audio_train[arr_po[i]])
        y_train2_audio.append(y_audio_train[arr_po[i]])
    for i in range(0,flod_po):
        x_train2.append(x_text_train[arr_po[i]])
        y_train2.append(y_text_train[arr_po[i]])
        x_train2_vision.append(x_vision_train[arr_po[i]])
        y_train2_vision.append(y_vision_train[arr_po[i]])
        x_train2_audio.append(x_audio_train[arr_po[i]])
        y_train2_audio.append(y_audio_train[arr_po[i]])
    for i in range(len_ne+3 * flod_po, 5*flod_po):
        x_train3.append(x_text_train[arr_po[i]])
        y_train3.append(y_text_train[arr_po[i]])
    for i in range(0,2 * flod_po):
        x_train3.append(x_text_train[arr_po[i]])
        y_train3.append(y_text_train[arr_po[i]])
    for i in range(4 * flod_po, 5* flod_po):
        x_train4.append(x_text_train[arr_po[i]])
        y_train4.append(y_text_train[arr_po[i]])
    for i in range(0,3 * flod_po):
        x_train4.append(x_text_train[arr_po[i]])
        y_train4.append(y_text_train[arr_po[i]])
    for i in range(0,4 * flod_po):
        x_train5.append(x_text_train[arr_po[i]])
        y_train5.append(y_text_train[arr_po[i]])

    return x_train1, y_train1, x_train2, y_train2, x_train3, y_train3, x_train4, y_train4, x_train5, y_train5, x_test1, y_test1, x_test2, y_test2, x_test3, y_test3, x_test4, y_test4, x_test5, y_test5,x_train2_vision,y_train2_vision,x_train2_audio,y_train2_audio,x_test2_vision,y_test2_vision,x_test2_audio,y_test2_audio










