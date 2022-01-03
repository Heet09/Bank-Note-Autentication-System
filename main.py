import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('BankNote_Authentication.csv')
# print(df.head())

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# print(f'x_train : {x_train.shape}')
# print(f'y_train : {y.shape}')
# print(f'x_test : {x_test.shape}')
# print(f'y_test : {y_test.shape}')

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
score = accuracy_score(y_pred, y_test)

print(score)

pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
