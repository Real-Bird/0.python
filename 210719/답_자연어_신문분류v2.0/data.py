import pandas as pd
import option as opt
import os
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import preprocessing

op = opt.Options()
tokenizer = Tokenizer(num_words=1000)

def get_word_index() :    # 단어 인덱스를 구축
    for idx, dirnames in enumerate ( op.clean_data_dir ) :
            for filename in os.listdir(dirnames ): 
                    infilename = dirnames+"/"+filename
                    news  = pd.read_csv(infilename, header=None, encoding="euc-kr") 
                    tokenizer.fit_on_texts(news.iloc[:,0]) 
    #print("각 단어의 인덱스: \n", tokenizer.word_index) 
    return  tokenizer  

def make_onehot(t,  data ) :   # 원-핫 이진 벡터 표현
    one_hot_results  = t.texts_to_matrix(data)
    return one_hot_results 

def make_word_seq(t,  data ) :
    word_seq_results  = t.texts_to_sequences(data)
    word_seq_results = preprocessing.sequence.pad_sequences(word_seq_results, maxlen=op.max_len)
    return word_seq_results

def load_one_word_seq(t) :
    X_total = ""
    Y_total = ""

    for idx, dirnames in enumerate(op.clean_data_dir):
        for fidx, filename in enumerate(os.listdir(dirnames)):
            infilename = dirnames + "/" + filename
            news = pd.read_csv(infilename, header=None, encoding="euc-kr")
            if fidx == 0 and idx == 0:
                X_total = make_word_seq(t, news.iloc[:, 0])
                Y_total = news.iloc[:, 1]
            else:
                X_total = np.concatenate([X_total, make_word_seq(t, news.iloc[:, 0])])
                Y_total = np.concatenate([Y_total, news.iloc[:, 1]])

    return (X_total, Y_total)


def load_one_hot_data(t) :
    X_total = ""
    Y_total = "" 

    for idx, dirnames in enumerate ( op.clean_data_dir ) :
            for fidx, filename in enumerate ( os.listdir( dirnames ) ) : 
                    infilename = dirnames+"/"+filename 
                    news  = pd.read_csv(infilename, header=None, encoding="euc-kr")  
                    if fidx==0 and idx==0 : 
                        X_total = make_onehot(t, news.iloc[:,0])
                        Y_total = news.iloc[:,1]
                    else :
                        X_total =  np.concatenate([X_total,  make_onehot(t, news.iloc[:,0]) ])
                        Y_total =  np.concatenate([Y_total,  news.iloc[:,1] ])

    return (X_total, Y_total) 

