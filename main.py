from google.colab import drive
drive.mount('/drive')
%cd /drive

import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Activation,Dropout,Embedding,SpatialDropout1D
from tensorflow.keras.layers import LSTM,GRU,Bidirectional,BatchNormalization,Conv1D,GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

data = pd.read_csv("/drive/My Drive/Colab Notebooks/LSTM,GRU,Bİ-LSTM,CNN/consumer/consumer_complaints.csv")

data.loc[data['product'] == 'Credit reporting', 'product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
data.loc[data['product'] == 'Credit card', 'product'] = 'Credit card or prepaid card'
data.loc[data['product'] == 'Payday loan', 'product'] = 'Payday loan, title loan, or personal loan'
data.loc[data['product'] == 'Virtual currency', 'product'] = 'Money transfer, virtual currency, or money service'

def print_plot(index):
    overwiev= data[data.index == index][['issue', 'product']].values[0]
    if len(overwiev) > 0:
        print('issue:',overwiev[0])
        print('Product:', overwiev[1])
print_plot(20)


max_nb_words = 50000
max_sequence_lenght = 250
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['issue'].values)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))


x_data = tokenizer.texts_to_sequences(data['issue'].values)
x_data = pad_sequences(x_data, maxlen=max_sequence_lenght)

print('Shape of data tensor:', x_data.shape,'\n')
print(x_data[:5],'\n')
print(data['issue'][:5])


y_data = pd.get_dummies(data['product']).values
print('Shape of label tensor:', y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.10, random_state = 42)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)

filters = 250
kernel_size = 3

model = Sequential()

model.add(Embedding(max_nb_words, embedding_dim, input_length=x_data.shape[1]))
model.add(Dropout(0.2))
model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(250))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(11))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

epochs = 5
batch_size = 512

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    validation_data = (x_test,y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

plt.title('Loss')
plt.plot(history.history['loss'],label= 'train loss')
plt.plot(history.history['val_loss'],label= 'validation loss')
plt.legend()
plt.show()


plt.title('Accuracy')
plt.plot(history.history['accuracy'],label= 'traininh acc')
plt.plot(history.history['val_accuracy'],label= 'validation acc')
plt.legend()
plt.show()

