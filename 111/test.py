import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import pickle
from util import Dataset


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.txt', '.xml']
app.config['UPLOAD_PATH'] = 'uploads'


@app.route('/')
def index():
    return render_template('index.html', s=0)


@app.route('/', methods=['POST'])
def upload_files():
    # 它是一个不是一个基准变量
    pre = {}
    name = {}
    uploaded_files = request.files.getlist('file')
    k=0
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        name[k] = filename
        k=k+1
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            file.save(os.path.join(app.config['UPLOAD_PATH'],filename))

    file = open('D:/pycharm/project/抑郁检测/ufo-model.pkl', 'rb')
    model = pickle.load(file)
    file.close()
    x_test = Dataset.AllFile("D:/pycharm/project/抑郁检测/111/uploads")
    test = model.predict(x_test)


    for i in range(0, len(test)):

        if test[i] == 'ne':
            # print("非抑郁")
            a = "抑郁"
            pre[name[i]] = a

        if test[i] == 'po':
            # print("抑郁")
            b = "非抑郁"
            pre[name[i]] = b
    print(pre)

    return render_template('index.html', s=1,results=pre)


if __name__ == "__main__":
    app.run(debug=True)
