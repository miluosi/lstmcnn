from sklearn.externals import joblib
from keras.models import load_model
import pandas as pd
from keras.preprocessing import sequence
import numpy as np
import jieba
import re
def lstmcnn_5_predict(x):
    jieba_res=fenci(x)
    table=pd.DataFrame(columns=['roue'],index=[0])
    table['roue']=jieba_res
    #导入训练完的词向量
    tok_1=joblib.load(r'C:\Users\86139\Desktop\python程序\文本数据处理\tok.pkl_100')
    test_seq=tok_1.texts_to_sequences(table['roue'])
    #序列化,长度一致
    test_seq_mat=sequence.pad_sequences(test_seq,maxlen=100)
    sequences=test_seq_mat
    #导入训练完的模型
    model = load_model(r'C:\Users\86139\Desktop\python程序\文本数据处理\Tezt_cnn_2_5.h5')
    p=model.predict(test_seq_mat)
    result = np.argmax(p)
    if(result==0):

       # print(model.predict(sequences))
        return model.predict(sequences)
        
    if(result==1):
        
        #print(model.predict(sequences))
        return model.predict(sequences)
    if(result==2):
        
        #print(model.predict(sequences))
        return model.predict(sequences)
    if(result==3):
        return model.predict(sequences)
        #print(model.predict(sequences))
    
def fenci(x):
    data = re.sub("[A-Za-z0-9\！\%\[\]\,\。\?\·\，\“\”\、\：\】\【\）\（\》\《\＂]", "", x)
    seg_list = jieba.cut(data)
    jieba_res=" ".join(seg_list)
    return jieba_res

a=lstmcnn_5_predict("哈哈哈！")
print(a)
print(a[0,0])