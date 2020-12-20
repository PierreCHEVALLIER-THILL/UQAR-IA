import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator as IDG
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
%matplotlib inline
from keras.preprocessing.image import load_img,img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint

with open("../input/finalimport/model.json") as json_file:
    base_model = model_from_json(json_file.read())
    base_model.load_weights("../input/finalimport/model_weights.h5")

b_path = "../input/facial-expression-recognition/test/test/"
batch_size = 128
epochs = 200
pic_size = 48

data = IDG().flow_from_directory(b_path,target_size=(pic_size,pic_size),
                                       color_mode="grayscale",batch_size=batch_size,
                                       class_mode="categorical",shuffle=False)

n_classes = 7

plt.figure(0,figsize=(20,20))
cpt=0
for expression in os.listdir(b_path):
    for i in range(1,8):
        cpt += 1
        plt.subplot(7,8,cpt)
        img=load_img(b_path+expression+"/"+os.listdir(b_path+expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img,cmap='gray')
        plt.xlabel(os.listdir(b_path+expression)[i])
plt.tight_layout()
plt.show()

base_model.trainable = False

print(base_model.input);

inputs = keras.Input(shape=(48, 48, 1))
x = base_model(inputs, training=False)
outputs = keras.layers.Dense(7)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(data, epochs=epochs, callbacks=callbacks_list)

Y_pred = model.predict_generator(data,  data.n // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(data.classes, y_pred))
print('Classification Report')
target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(classification_report(data.classes, y_pred, target_names=target_names))




model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)