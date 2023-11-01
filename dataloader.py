import sys
import numpy as np
import pandas as pd
from preprocesser.clean_text import finalfunction
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
print(sys.argv)
dataset = sys.argv[1];
if dataset == "threeag":
    train_data = pd.read_csv("threeaggdata/threeaggtrain.csv")
    test_data = pd.read_csv("threeaggdata/threeaggtest.csv")
train_data = train_data.drop(["facebook_corpus_msr_1723796"],axis=1)
new_cols = {"Well said sonu..you have courage to stand against dadagiri of Muslims":"data","OAG" : "Label"}
train_data.rename(columns=new_cols,inplace=True)
train_data["CleanText"] = [finalfunction(text) for text in train_data["data"]]
train_data = train_data[train_data["CleanText"].apply(lambda x : len(x) > 0)]
train_data["ConvertedLabel"] = train_data["Label"].replace({"OAG" : 0,"CAG" : 1,"NAG" : 2})
train_data.to_csv("threeaggdata",index=False)
#Params
max_seq_length = 20;
embedding_dim = 100;
max_words = 10000;
X = train_data["CleanText"];
Y = train_data["ConvertedLabel"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=30,random_state=42)
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences_train = tok.texts_to_sequences(X_train)
sequences_test = tok.texts_to_sequences(X_test)
sequences_train_padded = pad_sequences(sequences_train,max_seq_length,padding="post")
sequences_test_padded = pad_sequences(sequences_test,max_seq_length,padding="post")
y_train_conv = to_categorical(Y_train)
y_test_conv = to_categorical(Y_test)
print(len(tok.word_index))
