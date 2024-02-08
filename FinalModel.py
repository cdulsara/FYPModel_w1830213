import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,BatchNormalization
from tensorflow.keras.layers import GlobalAvgPool2D, Dense
from keras.preprocessing import image

def create_dataframe(data_path):
    df = []
    for c in os.listdir(data_path):
        class_folder = os.path.join(data_path, c)
        for f in os.listdir(class_folder):
            f_path = os.path.join(class_folder, f)
            if f_path.endswith('JPG'):
                df.append([f_path, c])
    return pd.DataFrame(df, columns=('filename', 'class'))

# constants to be used in creating dataframes
IMG_DIM = 256
DATA_PATH = 'data/'
CLASSES = sorted(['TRI5001', 'TRI5002', 'TRI5003','TRI5004'])
print(CLASSES)

# creating dataframes
df = create_dataframe(os.path.join(DATA_PATH, 'train'))
df_test = create_dataframe(os.path.join(DATA_PATH, 'test'))

# splitting the training dataframe into train and val
df_train, df_val = train_test_split(df, test_size=0.30, random_state=0)

# plotting the number of images of each class in the data
sns.countplot(x = df["class"])
plt.xticks(rotation = 0);

train_dir = 'data/train/'
plt.figure(figsize=(14,22))
i = 1
for expression in os.listdir(train_dir):
    img = load_img((train_dir + expression +'/'+ os.listdir(train_dir + expression)[1]))
    plt.subplot(1,7,i)
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')
    i += 1
plt.show()

df_train

train_gen = ImageDataGenerator().flow_from_dataframe(
    df_train,
    target_size=(IMG_DIM, IMG_DIM),
    classes=CLASSES,
    color_mode='grayscale',
    width_shift_range = 0.3, 
    height_shift_range = 0.3, 
    rotation_range=30,
    horizontal_flip = True, 
    rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    shuffle=False,
)

val_gen = ImageDataGenerator().flow_from_dataframe(
    df_val,
    target_size=(IMG_DIM, IMG_DIM),
    classes=CLASSES,
    color_mode='grayscale', 
    width_shift_range = 0.3, 
    height_shift_range = 0.3, 
    rotation_range=30,
    horizontal_flip = True, 
    rescale = 1./255,  
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    shuffle=False,
)

test_gen = ImageDataGenerator().flow_from_dataframe(
    df_test,
    target_size=(IMG_DIM, IMG_DIM),
    classes=CLASSES,
    color_mode='grayscale', 
    width_shift_range = 0.3,
    height_shift_range = 0.3, 
    rotation_range=30,
    horizontal_flip = True,
    rescale = 1./255,  
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    shuffle=False,
)

train_gen.class_indices

# plotting the number of images of each class in the data
sns.countplot(x = df["class"])
plt.xticks(rotation = 0);

img = train_gen[0][0][1]
img.shape

# Initialize the CNN model
model = Sequential()

# Add Convolutional layers
model.add(Conv2D(32, (3, 3),padding='same', input_shape=(IMG_DIM, IMG_DIM, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3),padding='same', activation='relu',name='last_conv'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# Flatten the output before feeding into the fully connected layers
model.add(Flatten())

# Add Dense layers for classification
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax')) 

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

# make a check point that saves the best epoch
checkpoint_path = 'working/my_cnn_model.h5'

checkpoint = ModelCheckpoint(checkpoint_path,
                            save_weights_only=True,
                            save_best_only=True,
                            verbose=1,
                            mode='min')

# train the model in 20 epochs
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[checkpoint],
    shuffle=True
)

# Save the model in HDF5 format
model.save('working/my_cnn_model.h5')

# plot the graphs that shows the progress of the training of each epoch and compare the training accuracy with the validation accuracy

fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12,4)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

y_pred = model.predict(train_gen)
y_pred = np.argmax(y_pred, axis=1)
class_labels = train_gen.class_indices
class_labels = {v:k for k,v in class_labels.items()}

from sklearn.metrics import classification_report, confusion_matrix
cm_train = confusion_matrix(train_gen.classes, y_pred)
print('Confusion Matrix')
print(cm_train)
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(train_gen.classes, y_pred, target_names=target_names))

plt.figure(figsize=(8,8))
plt.imshow(cm_train, interpolation='nearest')
plt.colorbar()
tick_mark = np.arange(len(target_names))
_ = plt.xticks(tick_mark, target_names, rotation=90)
_ = plt.yticks(tick_mark, target_names)

y_pred = model.predict(test_gen)
y_pred = np.argmax(y_pred, axis=1)
class_labels = test_gen.class_indices
class_labels = {v:k for k,v in class_labels.items()}

from sklearn.metrics import classification_report, confusion_matrix
cm_test = confusion_matrix(test_gen.classes, y_pred)
print('Confusion Matrix')
print(cm_train)
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(test_gen.classes, y_pred, target_names=target_names))

plt.figure(figsize=(8,8))
plt.imshow(cm_test, interpolation='nearest')
plt.colorbar()
tick_mark = np.arange(len(target_names))
_ = plt.xticks(tick_mark, target_names, rotation=90)
_ = plt.yticks(tick_mark, target_names)

# evaluate the model on the testing images
print(checkpoint_path)
model.load_weights(checkpoint_path)
model.evaluate(test_gen)

image = cv2.imread('data/test/TRI5002/IMG_1451.JPG',cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image,(256,256),interpolation=cv2.INTER_LINEAR)
img=np.array(image)
img=img.reshape(1,256,256,1)
predict_x=model.predict(img) 
result=np.argmax(predict_x)
print(CLASSES[result])
plt.imshow(image,cmap='gray')
plt.show()
