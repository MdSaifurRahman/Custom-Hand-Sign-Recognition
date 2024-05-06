import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Pad or truncate sequences to a fixed length
max_sequence_length = 100  # Adjust this value based on your data
padded_data = pad_sequences(data_dict['data'], maxlen=max_sequence_length, padding='post', truncating='post', dtype='float32')

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
