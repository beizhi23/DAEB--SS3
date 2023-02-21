import os

from flask import Flask, render_template, request, send_from_directory
import pickle

from util import Dataset
app: Flask = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"


@app.route("/")
def index():
    return render_template("upload.html")


@app.route('/uploads', methods=['POST', 'GET'])
def uploader():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
        # 获取文件对象
        f_obj = request.files['file']
        # print(f_obj.filename)
        f_obj.save(os.path.join(app.config['UPLOAD_FOLDER'], f_obj.filename))
        # 加载模型
        file = open('D:/pycharm/project/抑郁检测/ufo-model.pkl', 'rb')
        model = pickle.load(file)
        file.close()
        x_test = Dataset.AllFile(
            "D:/pycharm/project/抑郁检测/fileupload/fileupload/uploads")
        test=model.predict(x_test)
        pre=[]
        for i in range(0,len(test)):
            if test[i]=='ne':
                #print("非抑郁")
                a=["非抑郁"]
                pre=' '.join(a)

            if test[i]=='po':
                #print("抑郁")
                b=["抑郁"]
                pre=' '.join(b)

        return pre


@app.route('/upload/<filename>')
def upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
