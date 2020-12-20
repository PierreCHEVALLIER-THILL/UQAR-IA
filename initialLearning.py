#lib calcule
import numpy as np
#lib double tableau (avec label)
import pandas as pd
#Lib system (ex parcourrir ordinateur)
import os
#Afficher des courbes

import matplotlib.pyplot as plt
#lib graphique (schema/graph)
import seaborn as sns

#lib pour faire traiter image en amont
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import ImageDataGenerator as IDG

#Permet de faire une sauvegarde
from keras.callbacks import ModelCheckpoint

#Permet de gérer les couches
from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D

#Permet de gérer le model
from keras.models import Model,Sequential

#Permet optimisation
from keras.optimizers import Adam,SGD,RMSprop

from sklearn.metrics import classification_report, confusion_matrix

#Pour que la lib ouvre pas de fenetre et travail dans le terminal (redirige sortie)
%matplotlib inline

#Taille en pixel des images
pic_size = 48

#Path to Dataset
b_path = "../input/face-expression-recognition-dataset/images/"


#Va prendre 128 photo / photo pour update le model, (ex 200 photo en entré et que batch_size = 5, et je fais 1000 epoch ca veut dire que dataSet sera divisé par 40 (ce qui donne 40 batches/paquet))
# dans chacun de ces batches j'aurais 5 photos. Pourquoi definitons des paquets. Tous les 5 photo ou tous les paquets(a chaque paquet) le Model est update.
# A une epoch implique 40 paquetes/batches donc au sein d'une epoch 40 update. Et donc avec les 1000 epoch on traverse dataSET 1000 fois ca fait un total de 40 000 batches/paquets qui sont lu pendant l'entrainement
#  

batch_size = 128 #Valeur correct qui se retrouve souvent dans les projet IA, multiple de 2
epochs = 200

#IDG() Permet de pretraiter/formater en amont les images en objet en python/tenser (converti img en tableau de pixel)
#IDG(), n'applique aucun traitement. flow_from_directory() on recupère les images , on précise, la taille en pixel:pic_size
    # la color (noir et blanc), batch_size ?
    # class_mode: entrainer le model par categorie et pas en binaire (binary/categorical = 7 emotion)

train_gen = IDG().flow_from_directory(b_path+"train",target_size=(pic_size,pic_size)
                                          ,color_mode="grayscale",batch_size=batch_size,
                                          class_mode="categorical",shuffle=True)
val_gen = IDG().flow_from_directory(b_path+"validation",target_size=(pic_size,pic_size),
                                       color_mode="grayscale",batch_size=batch_size,
                                       class_mode="categorical",shuffle=False)


n_classes = 7 #Nombre de label/emotions


# Creation du model avec detail de chaque layers


#layer 1 Prend une image de 48 par 48
model = Sequential() # initiaalisation du model
model.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,1))) #1arg je ne sais pas?/2nd parametre taille de la fenetre de convolution. Padding (Gestion en fonction de la taille de la fenetre de convolution, si c'est décalé)
model.add(BatchNormalization()) # je ne sais pas Apparement amelioration
model.add(Activation('relu')) # Permet de supprimer les valeur negative du model pour pas generer d'erreur et augmenter la precision
model.add(MaxPooling2D(pool_size=(2,2))) #Concept qui permet de prendre les meilleurs valeur des la taille de la fenetre de convolution
model.add(Dropout(0.25)) # Apparement supprime/ignore aléatoirement des neurones pendant la phase de training

#layer 2 Prend en entré la sortie de la layers 1
model.add(Conv2D(128,(5,5),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))# Permet de supprimer les valeur negative du model pour pas generer d'erreur et augmenter la precision
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # Permet de reduire le nombre de layer/couche tableau de 2 a 1 une dimension
#FC Layer 1 Genere le vote (comme dans le tuto) Permettant de determiner l'émotion (celui-ci s'affine pendant l'entrainement et deviens plus précis)
model.add(Dense(256)) # applique le poids (qui permettait de donner plus dimpacte a une caractéristique(output layer2)).
model.add(BatchNormalization()) #Optimisation
model.add(Activation('relu'))# Permet de supprimer les valeur negative du model pour pas generer d'erreur et augmenter la precision
model.add(Dropout(0.25))
#FC Layer 2
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu')) # Permet de supprimer les valeur negative du model pour pas generer d'erreur et augmenter la precision
model.add(Dropout(0.25))

model.add(Dense(n_classes,activation='softmax')) # Derniere etapes Effectue le vote pour toute les emotions
opt = Adam(lr = 0.001) # learning rate, plus il est proche de 1 plus je vais faire de l'overfiting (vitesse d'apprentissage, donc tres mauvais durant la validation) et si proche de 0, cas inverse.
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy']) # je ne sais pas 
model.summary() # je ne sais pas 

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=train_gen,
                                steps_per_epoch=train_gen.n//train_gen.batch_size,
                                epochs=epochs,
                                validation_data = val_gen,
                                validation_steps = val_gen.n//val_gen.batch_size,
                                callbacks=callbacks_list
                                ) # Fonction principal qui entraine le model en fonction des du dataset qui traverse les layers

Y_pred = model.predict_generator(val_gen,  val_gen.n // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(val_gen.classes, y_pred))
print('Classification Report')
target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(classification_report(val_gen.classes, y_pred, target_names=target_names))

def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    


    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()
 
# print best epoch with best accuracy on validation

def get_best_epcoh(history):
    valid_acc = history.history['val_accuracy']
    best_epoch = valid_acc.index(max(valid_acc)) + 1
    best_acc =  max(valid_acc)
    print('Best Validation Accuracy Score {:0.5f}, is for epoch {}'.format( best_acc, best_epoch))
    return best_epoch


plot_results(history)
best_epoch =get_best_epcoh(history)
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    
    

