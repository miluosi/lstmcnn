import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from keras.models import *
from keras.layers import *
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import jieba
data =pd.read_excel(r'C:\Users\86139\Desktop\review_4_cut.xlsx')
data['review'] = data['review'].astype(str)
data['fenci'] = data['fenci'].astype(str)
#训练集
train_0=data[data['label']==0][:50000]
train_1=data[data['label']==1][:50000]
train_2=data[data['label']==2][:50000]
train_3=data[data['label']==3][:50000]
train= pd.concat([train_0,train_1,train_2,train_3],ignore_index=True)
#稳定集
val_0=data[data['label']==0][50000:51000]
val_1=data[data['label']==1][50000:51000]
val_2=data[data['label']==2][50000:51000]
val_3=data[data['label']==3][50000:51000]
val=pd.concat([val_0,val_1,val_2,val_3],ignore_index=True)
#测试集
test_0=data[data['label']==0][51000:51500]
test_1=data[data['label']==1][51000:51500]
test_2=data[data['label']==2][51000:51500]
test_3=data[data['label']==3][51000:51500]
test=pd.concat([test_0,test_1,test_2,test_3],ignore_index=True)
data_1=pd.concat([train,val,test],ignore_index=True)
max_len=100
max_words=5000 #最大词语数量
tok_1=Tokenizer(num_words=max_words)#序列化
tok_1.fit_on_texts(data_1.fenci)
train_seq=tok_1.texts_to_sequences(train.fenci)
val_seq=tok_1.texts_to_sequences(val.fenci)
test_seq=tok_1.texts_to_sequences(test.fenci)
#序列化,长度一致
train_seq_mat=sequence.pad_sequences(train_seq,maxlen=max_len)
val_seq_mat=sequence.pad_sequences(val_seq,maxlen=max_len)
test_seq_mat=sequence.pad_sequences(test_seq,maxlen=max_len)
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
train_y=train.label #标签结果
val_y=val.label
test_y=test.label #标签
le=LabelEncoder()#文本转化为数字
train_y=le.fit_transform(train_y).reshape(-1,1)
val_y=le.fit_transform(val_y).reshape(-1,1)
test_y=le.fit_transform(test_y).reshape(-1,1)
ohe=OneHotEncoder()
train_y=ohe.fit_transform(train_y).toarray()
val_y=ohe.fit_transform(val_y).toarray()
test_y=ohe.fit_transform(test_y).toarray()
def kong(x):
    return len(x)
data_1['num of fenci']=data_1['fenci'].apply(kong)
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
sns.countplot(data_1.label)
plt.xlabel('Label')
plt.xticks(size=10)
plt.title('评论数据数量预览')
plt.show()
print(data_1['num of fenci'].describe())
plt.hist(data_1['num of fenci'],bins=100)
plt.title('评论数据长度预览')

def build_MLP(optimeizer="adam",init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(32, input_dim=100, activation="relu",kernel_initializer=init))
    model.add(Dense(16, activation="relu",kernel_initializer=init))
    model.add(Dense(4, activation="softmax",kernel_initializer=init))  # 输出
    model.compile(loss="categorical_crossentropy", optimizer=optimeizer, metrics=["accuracy"])
    return model

def CNNchuan_model():
    model=Sequential()
    model.add(Embedding(input_dim=5001,output_dim=256,input_length=100))
    #卷积层有助于提取特征
    model.add(Conv1D(filters=80,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=80,kernel_size=4,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=80,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=100,activation='softmax'))
    model.add(Dense(units=63,activation='relu'))#循环卷积神经网络
    model.add(Dense(units=32,activation='softmax'))
    model.add(Dense(units=4,activation='softmax'))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    print(model.summary())
    return model

def CNNconcate():
        main_input = Input(shape=(100,), dtype='float64')
        # 词嵌入（使用预训练的词向量）
        embedder = Embedding(5000+1, 300, input_length=100, trainable=False)
        embed = embedder(main_input)
        # 词窗大小分别为3,4,5
        cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=10)(cnn1)
        cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=10)(cnn2)
        cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = MaxPooling1D(pool_size=10)(cnn3)
        # 合并三个模型的输出向量
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = Flatten()(cnn)
        drop = Dropout(0.2)(flat)
        relu_1=Dense(128, activation='relu')(drop)
        relu_2=Dense(64, activation='relu')(relu_1)
        relu_3=Dense(32, activation='relu')(relu_2)
        main_output = Dense(4, activation='softmax')(relu_3)
        model = Model(inputs=main_input, outputs=main_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

def TextCNN_model_2():
        main_input = Input(shape=(100,), dtype='float64')
        # 词嵌入（使用预训练的词向量）
        embedder = Embedding(5000+1, 256, input_length=100, trainable=False)
        embed = embedder(main_input)
        # 词窗大小分别为3,4,5,6,7
        cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=10)(cnn1)
        cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=10)(cnn2)
        cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = MaxPooling1D(pool_size=10)(cnn3)
        # 合并三个模型的输出向量
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat_1=LSTM(units=256)(cnn)
        flat = Flatten()(flat_1)
        relu_1=Dense(256, activation='relu')(flat)
        relu_2=Dense(128, activation='relu')(relu_1)
        relu_3=Dense(64, activation='relu')(relu_2)
        relu_4=Dense(32, activation='relu')(relu_3)
        main_output = Dense(4, activation='softmax')(relu_4)
        model = Model(inputs=main_input, outputs=main_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

def TextCNN_model_3():
        main_input = Input(shape=(100,), dtype='float64')
        # 词嵌入（使用预训练的词向量）
        embedder = Embedding(5000+1, 256, input_length=100, trainable=False)
        embed = embedder(main_input)
        # 词窗大小分别为3,4,5,6,7
        cnn1 = Conv1D(256, 3, padding='same', strides=2, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=10)(cnn1)
        cnn2 = Conv1D(256, 4, padding='same', strides=2, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=10)(cnn2)
        cnn3 = Conv1D(256, 5, padding='same', strides=2, activation='relu')(embed)
        cnn3 = MaxPooling1D(pool_size=10)(cnn3)
        cnn4 = Conv1D(256, 6, padding='same', strides=2, activation='relu')(embed)
        cnn4 = MaxPooling1D(pool_size=10)(cnn4)
        cnn5 = Conv1D(256, 7, padding='same', strides=2, activation='relu')(embed)
        cnn5 = MaxPooling1D(pool_size=10)(cnn5)
        cnn6 = Conv1D(256, 8, padding='same', strides=2, activation='relu')(embed)
        cnn6 = MaxPooling1D(pool_size=10)(cnn6)
        # 合并三个模型的输出向量
        cnn = concatenate([cnn1, cnn2, cnn3,cnn4,cnn5,cnn6], axis=-1)
        flat_1=LSTM(units=1000)(cnn)
        flat = Flatten()(flat_1)
        relu_1=Dense(256, activation='relu')(flat)
        relu_2=Dense(128, activation='relu')(relu_1)
        relu_3=Dense(64, activation='relu')(relu_2)
        relu_4=Dense(32, activation='relu')(relu_3)
        main_output = Dense(4, activation='softmax')(relu_4)
        model = Model(inputs=main_input, outputs=main_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model


def TextCNN_model_4():
    main_input = Input(shape=(100,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(5000 + 1, 256, input_length=100, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5,6,7
    cnn1 = Conv1D(256, 3, padding='same', strides=2, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=10)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=2, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=10)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=2, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=10)(cnn3)
    cnn4 = Conv1D(256, 6, padding='same', strides=2, activation='relu')(embed)
    cnn4 = MaxPooling1D(pool_size=10)(cnn4)
    cnn5 = Conv1D(256, 7, padding='same', strides=2, activation='relu')(embed)
    cnn5 = MaxPooling1D(pool_size=10)(cnn5)
    cnn6 = Conv1D(256, 8, padding='same', strides=2, activation='relu')(embed)
    cnn6 = MaxPooling1D(pool_size=10)(cnn6)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3, cnn4, cnn5, cnn6], axis=-1)
    lstm1 = LSTM(1000, dropout=0.1, recurrent_dropout=0.1)(cnn)
    lstm1 = Dense(16, activation='relu')(lstm1)
    lstm1 = Dropout(0.1)(lstm1)
    lstm2 = LSTM(1000, dropout=0.1, recurrent_dropout=0.1)(cnn)
    lstm2 = Dense(16, activation='relu')(lstm2)
    lstm2 = Dropout(0.1)(lstm2)
    lstm3 = LSTM(1000, dropout=0.1, recurrent_dropout=0.1)(cnn)
    lstm3 = Dense(16, activation='relu')(lstm3)
    lstm3 = Dropout(0.1)(lstm3)
    lstm4 = LSTM(1000, dropout=0.1, recurrent_dropout=0.1)(cnn)
    lstm4 = Dense(16, activation='relu')(lstm4)
    lstm4 = Dropout(0.1)(lstm4)
    lstm5 = LSTM(1000, dropout=0.1, recurrent_dropout=0.1)(cnn)
    lstm5 = Dense(16, activation='relu')(lstm5)
    lstm5 = Dropout(0.1)(lstm5)
    lstm6 = LSTM(1000, dropout=0.1, recurrent_dropout=0.1)(cnn)
    lstm6 = Dense(16, activation='relu')(lstm6)
    lstm6 = Dropout(0.1)(lstm6)
    merge_1 = concatenate([lstm1, lstm2], axis=-1)
    merge_2 = concatenate([lstm3, lstm4], axis=-1)
    merge_3 = concatenate([lstm5, lstm6], axis=-1)
    merge = concatenate([merge_1, merge_2, merge_3], axis=-1)
    flat = Flatten()(merge)
    relu_1 = Dense(256, activation='relu')(flat)
    relu_2 = Dense(128, activation='relu')(relu_1)
    relu_3 = Dense(64, activation='relu')(relu_2)
    relu_4 = Dense(32, activation='relu')(relu_3)
    main_output = Dense(4, activation='softmax')(relu_4)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
model=build_MLP()
history=model.fit(train_seq_mat,train_y,batch_size=1000,epochs=20,validation_data=(val_seq_mat,val_y))
scores=model.evaluate(train_seq_mat,train_y,verbose=0)#评分
print("%s %f" % (model.metrics_names[1],scores[1]*100))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train","validation"],loc="upper left")
plt.show()

#loss损失函数
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","validation"],loc="upper left")
plt.show()
model.save('MLP.h5')

model=CNNchuan_model()
history=model.fit(train_seq_mat,train_y,batch_size=1000,epochs=20,validation_data=(val_seq_mat,val_y))
scores=model.evaluate(train_seq_mat,train_y,verbose=0)#评分
print("%s %f" % (model.metrics_names[1],scores[1]*100))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train","validation"],loc="upper left")
plt.show()

#loss损失函数
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","validation"],loc="upper left")
plt.show()
model.save('CNNchuan.h5')
model=CNNconcate()
history=model.fit(train_seq_mat,train_y,batch_size=1000,epochs=20,validation_data=(val_seq_mat,val_y))
scores=model.evaluate(train_seq_mat,train_y,verbose=0)#评分
print("%s %f" % (model.metrics_names[1],scores[1]*100))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train","validation"],loc="upper left")
plt.show()

#loss损失函数
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","validation"],loc="upper left")
plt.show()
model.save('concate_1.h5')


model=TextCNN_model_2()
history=model.fit(train_seq_mat,train_y,batch_size=1000,epochs=20,validation_data=(val_seq_mat,val_y))
scores=model.evaluate(train_seq_mat,train_y,verbose=0)#评分
print("%s %f" % (model.metrics_names[1],scores[1]*100))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train","validation"],loc="upper left")
plt.show()

#loss损失函数
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","validation"],loc="upper left")
plt.show()
model.save('Textcnn_2_2.h5')


model=TextCNN_model_3()
history=model.fit(train_seq_mat,train_y,batch_size=1000,epochs=20,validation_data=(val_seq_mat,val_y))
scores=model.evaluate(train_seq_mat,train_y,verbose=0)#评分
print("%s %f" % (model.metrics_names[1],scores[1]*100))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train","validation"],loc="upper left")
plt.show()

#loss损失函数
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","validation"],loc="upper left")
plt.show()
model.save('Textcnn_2_3.h5')

model=TextCNN_model_4()
history=model.fit(train_seq_mat,train_y,batch_size=100,epochs=10,validation_data=(val_seq_mat,val_y))
scores=model.evaluate(train_seq_mat,train_y,verbose=0)#评分
print("%s %f" % (model.metrics_names[1],scores[1]*100))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train","validation"],loc="upper left")
plt.show()

#loss损失函数
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train","validation"],loc="upper left")
plt.show()
model.save('Textcnn_2_5.h5')

def fenci(x):
    import re
    data = re.sub("[A-Za-z0-9\！\%\[\]\,\。\?\·\，\“\”\、\：\】\【\）\（\》\《\＂]", "", x)
    seg_list = jieba.cut(data)
    jieba_res=" ".join(seg_list)
    return jieba_res

def lstmcnn_5_predict(x):
    from sklearn.externals import joblib
    from keras.models import load_model
    jieba_res=fenci(x)
    table=pd.DataFrame(columns=['route'],index=[0])
    table['route']=jieba_res
    #导入训练完的词向量
    tok_1=joblib.load('tok.pkl_100')
    test_seq=tok_1.texts_to_sequences(table['route'])
    #序列化,长度一致
    test_seq_mat=sequence.pad_sequences(test_seq,maxlen=max_len)
    sequences=test_seq_mat
    model = load_model('Textcnn_2_5.h5')
    #导入训练完的模型
    p=model.predict(test_seq_mat)
    result = np.argmax(p)
    if(result==0):
        print(sequences)
        print(model.predict(sequences))
        print("喜悦")
    if(result==1):
        print(sequences)
        print(model.predict(sequences))
        print("愤怒")
    if(result==2):
        print(sequences)
        print(model.predict(sequences))
        print("厌恶")
    if(result==3):
        print(sequences)
        print(model.predict(sequences))
        print("低落")