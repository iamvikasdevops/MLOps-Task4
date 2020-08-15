from keras.datasets import mnist
dataset = mnist.load_data('mymnist.db')
train , test = dataset
X_train , y_train = train
X_test , y_test = test
#img1 = X_train[7]
#import cv2
#import matplotlib.pyplot as plt
#plt.imshow(img1 , cmap='gray')
#img1_1d = img1.reshape(28*28)
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
     metrics=['accuracy']
     )
h = model.fit(X_train, y_train_cat, epochs=10)
scores = model.evaluate(X_test, y_test_cat, verbose=0)
print(scores[1]*100 , file = open("/keras/output.txt","a"))
if scores[1]*100>=90:
     model.save("/keras/mnist.h5")