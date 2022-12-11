# Tensorflow & Keras tutorial from Udemy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Hide tensorflow warmings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv('fake_reg.csv')

print(df.head())

#fg = sns.pairplot(df)
#plt.show()

from sklearn.model_selection import train_test_split

X = df.drop('price', axis=1).values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Method one add all layers in a list
'''
model = Sequential([Dense(units=4, activation='relu'),
                    Dense(units=2, activation='relu'),
                    Dense(units=1)])
'''
# Method 2, create empty seqwntial model and add layers one by one
model = Sequential()

# Add 'hidden' layers
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

# Last layer should be a single neuron, because it's trying to predict the price
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')
model.fit(x=X_train, y=y_train, epochs=250, verbose=0)

loss_df = pd.DataFrame(model.history.history)
#loss_df.plot()
#plt.show()

print(model.evaluate(X_test, y_test, verbose=0))
print(model.evaluate(X_train, y_train, verbose=0))

test_preds = model.predict(X_test)
test_preds = pd.Series(test_preds.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df = pd.concat([pred_df, test_preds],axis=1)
pred_df.columns = ['Test True Y','Model Predictions']

from sklearn.metrics import mean_squared_error
print('RMS error = {}'.format(mean_squared_error(pred_df['Test True Y'],pred_df['Model Predictions'])**0.5))

# Predict price from new item with features
new_item = [[998,1000]]

# Remember to scale the features
new_item = scaler.transform(new_item)
print(model.predict(new_item))

# To save a model
from tensorflow.keras.models import load_model
model.save('my_item_model.h5')

later_model = load_model('my_item_model.h5')