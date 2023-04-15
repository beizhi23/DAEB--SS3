import pickle
from Document_internal_data_enhancement import DAEB
import util


X_DATA ,Y_DATA = util.Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data)/kflod")
DAEB.DAEB_bagging_fit(X_DATA,Y_DATA,2/3,1,3,0.6)


model_filename = 'ufo-model1.pkl'  # 设定文件名
pickle.dump(DAEB, open(model_filename, 'wb'))  # 对模型进行“腌制”
model = pickle.load(open('ufo-model1.pkl', 'rb'))  # 加载“腌制”好的模型
print(model.predict(['depression']))  # 测试模型