print('hello') 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


import json
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader

# from FVQADataset import FVQARelationClassifierDataset
# from model import RelationClassifierLSTM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import cv2 as cv
import numpy as np

from torch.utils.data import Dataset
from string import punctuation
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt



class FVQARelationClassifierDataset(Dataset):
    def __init__(self, question_features, relation_features):
        super().__init__()
        self.question_features = question_features
        self.relation_features = relation_features

    def __getitem__(self, idx: int):
        return self.question_features[idx], self.relation_features[idx]

    def __len__(self) -> int:
        return len(self.question_features)

fvqa_path = '/home/seecs/afzaalhussain/thesis/fvqa-data/new_dataset_release'
path2 = os.path.join(fvqa_path, 'all_fact_triples_release.json')
with open(path2, encoding="utf8") as f:
    facts_data_json = json.load(f)

path = os.path.join(fvqa_path, 'all_qs_dict_release.json')
with open(path, encoding="utf8") as f:
    question_data_json = json.load(f)

relations_list = list()
questions_list = list()
relatoins_list_extra = list()

count = 1
for question in question_data_json:
    question_ = question_data_json[question]['question']
    kb_src = question_data_json[question]['kb_source'].replace("'", "")
    relation_ = facts_data_json[question_data_json[question]['fact'][0]]['r']
    if 'webchild' in kb_src:
        relations_list.append('Comparative')
    elif 'dbpedia' in kb_src:
        relations_list.append('Category')
    else:
        relation_ =  relation_.replace("/r/", "")
        relations_list.append(relation_)
    questions_list.append(question_)
    
# for fact in facts_data_json:
#     kb =  facts_data_json[fact]['KB']
#     relation_ = facts_data_json[fact]['r']
#     if kb == 'webchild':
#         relations_list.append('Comparative')
#     elif kb == 'dbpedia':
#         relations_list.append('Category')
#     else:
#         relation_ =  relation_.replace("/r/", "")
#         relations_list.append(relation_)

all_relations_text = ' '.join([c for c in relations_list if c not in punctuation])
relations_words = all_relations_text.split()
relations_count_words = Counter(relations_words)
relations_total_words = len(relations_words)
relations_sorted_words = relations_count_words.most_common(relations_total_words)
relations_vocab_to_int = {w:i for i, (w,c) in enumerate(relations_sorted_words)}
relations_int_to_vocab = {i:w for i, (w,c) in enumerate(relations_sorted_words)}



questions_all_text = ' '.join([c for c in questions_list if c not in punctuation])
# create a list of words
questions_words = questions_all_text.split()
# Count all the words using Counter Method
questions_count_words = Counter(questions_words)
questions_total_words = len(questions_words)
questions_sorted_words = questions_count_words.most_common(questions_total_words)
questions_vocab_to_int = {w:i+1 for i, (w,c) in enumerate(questions_sorted_words)}

questions_int = []
relations_int = []
relations_int = [relations_vocab_to_int[w] for w in relations_list]
for question in questions_list:
    word_int = [questions_vocab_to_int[w] for w in question.split()]
    questions_int.append(word_int)


relations_features = np.array(relations_int)
relation_onehot_encoding = to_categorical(relations_features, num_classes=13)

data = {'questions':questions_list, 'relation':relations_list, 'relation_index': relations_features,
        'relation_onhot':relation_onehot_encoding.tolist()}
# data
data = pd.DataFrame(data)
data.relation.value_counts()


n_most_common_words = 8000
max_len = 25
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['questions'].values)
sequences = tokenizer.texts_to_sequences(data['questions'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(X , relation_onehot_encoding, test_size=0.25, random_state=42)

epochs = 40
emb_dim = 128
batch_size = 256
relation_onehot_encoding[:2]

import tensorflow as tf
checkpoint_path = "training_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))
def create_model():   
    model = Sequential()
    model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.7))
    model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
    model.add(Dense(13, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model
model = create_model()

# print(model.summary())

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001),cp_callback ])


# Create a basic model instance
# model1 = create_model()
# Loads the weights
# model1.load_weights(checkpoint_path)

# # Re-evaluate the model
# accr = model1.evaluate(X_test,y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# accr = model.evaluate(X_test,y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))



# import matplotlib.pyplot as plt

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()



labels = data['relation'].value_counts().index
predictions = []
count = 1
data_question = data[data.relation == 'AtLocation']

# # print(data_question)
# for ques in data['questions']:  
#     ques = [ques]
#     seq = tokenizer.texts_to_sequences(ques)
#     padded = pad_sequences(seq, maxlen=max_len)
#     pred = model.predict(padded)
#     top3Pred = labels[(-pred).argsort()[0][:3]]
#     predictions.append(top3Pred)
#     print(count,') ', ques);
#     print(top3Pred.values)
#     print('=============================================')
#     count+=1

# data['predictions'] = predictions


df = pd.DataFrame(question_data_json)
df = df.transpose()
count = 1
labels = data['relation'].value_counts().index
predictions = []

detections_path = '/home/seecs/afzaalhussain/thesis/fvqa-data/new_dataset_release/scg-detections-coco-imagenet-scg-places365/scg-detections/'
detections_file_path = [os.path.join(detections_path, x) for x in os.listdir(detections_path)]


# while (num_images >= 0):
with_questions = 1
without_questions = 1


# merged_ques_facts_df = pd.read_pickle("merged_ques_facts_df.pkl")  


# import re
# predictions = []
# count = 0
# import time
# for key, row in merged_ques_facts_df.iterrows():
#     # model1.load_weights(checkpoint_path)
#     ques = [row['question']]
#     seq = tokenizer.texts_to_sequences(ques)
#     padded = pad_sequences(seq, maxlen=max_len)
#     # print(seq) 
#     pred = model1.predict(padded)
#     top3Pred = labels[(-pred).argsort()[0][:3]].copy()
#     # print(ques) 
#     # print(top3Pred)
#     predictions.append(top3Pred)
#     # time.sleep(1)
#     # if count == 5:
#     #     break
#     # count+=1
    
# merged_ques_facts_df['predicted_relations'] = predictions
# merged_ques_facts_df.to_pickle('merged_ques_facts_df.pkl')



# for detection_file in detections_file_path:
#     f = open(detection_file)
#     detections_json = json.load(f)

#     if 'all_detections' in detection_file:
#         continue
    
#     sub_df = df[df['img_file'] == detections_json['filename']]
#     if len(sub_df) == 0:       
#         print(without_questions ,') empty questions for' , detections_json['filename'])
#         without_questions+=1 
#         continue
#     else:
#         # print(with_questions, ') questions for' , detections_json['filename'])
#         # print(sub_df)
#         detections_json.pop('questions', None)
#         dicts = {}
#         for index, row in sub_df.iterrows():
#             ques = [row.question]
#             seq = tokenizer.texts_to_sequences(ques)
#             padded = pad_sequences(seq, maxlen=max_len)
#             pred = model.predict(padded)
#             top3Pred = labels[(-pred).argsort()[0][:3]]
#             predictions.append(top3Pred)
#             # print(row.question_id,') ', row['img_file'] , ' - ', ques);
#             # print(top3Pred.values)

#             ques_dic = {'question': ques, 'predicted_relations': top3Pred.values.tolist()}
#             dicts[row.question_id] =  ques_dic
#             # print(dicts[row.question_id])

#         detections_json['questions'] =  dicts
#         # detections_json['questions'] = ques_predict_rel
#         print(detections_json['questions'])
#         print('=========================================================')
#     with open(detection_file, 'w') as f:
#         json.dump(detections_json, f)

#     # print(count)
#     # print('detection_file :', detection_file)
#     # print('image_file_name ', detections_json['filename'])
#     # print()
#     # count += 1

# print('without questions count images = ', without_questions)
# print('with questions count images = ', with_questions)
# for index, row in df.iterrows():
#     if count >= 15:
#         break
#     count += 1
#     ques = [row.question]
#     seq = tokenizer.texts_to_sequences(ques)
#     padded = pad_sequences(seq, maxlen=max_len)
#     pred = model.predict(padded)
#     top3Pred = labels[(-pred).argsort()[0][:3]]
#     predictions.append(top3Pred)
#     print(row.question_id,') ', row['img_file'] , ' - ', ques);
#     print(top3Pred.values)
#     print('=============================================')


	




# txt = ["What object in this image is found in a shell?"]
# seq = tokenizer.texts_to_sequences(txt)
# print(seq)
# padded = pad_sequences(seq, maxlen=max_len)
# print(padded)
# pred = model.predict(padded)
# # labels = ['entertainment', 'bussiness', 'science/tech', 'health']
# labels = data['relation'].value_counts().index
# # print(labels)
# # print(pred, labels[np.argmax(pred)])
# print(labels[(-pred).argsort()[0][:3]])
# print(pred)
# print(len(pred[0]))