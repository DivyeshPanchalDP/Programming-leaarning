from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras.models import Sequential,load_model
import os
import tensorflow as tf




model= Sequential()
model.add(Conv2D(32,(3,3),input_shape=(256,256,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units=150,activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(units=6,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=12.,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.15,
                                 horizontal_flip=True)
val_datagen= ImageDataGenerator(rescale=1./255)
test_datagen= ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('images/train',
                                               target_size=(256,256),
                                               color_mode='grayscale',
                                               batch_size= 8,
                                               classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                               class_mode='categorical')

val_datagen=val_datagen.flow_from_directory('images/val',
                                            target_size=(256,256),
                                            color_mode='grayscale',
                                            batch_size=8,
                                            classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                            class_mode='categorical')

test_gen=test_datagen.flow_from_directory('images/test',
                                            target_size=(256,256),
                                            color_mode='grayscale',
                                            batch_size=8,
                                            classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                            class_mode='categorical')
callbacks_list=[EarlyStopping(monitor='val_loss',patience='10'),
                ModelCheckpoint(filepath='model_6cat_2.h6',monitor='val_loss',save_best_only=True)]

os.environ["CUDA_VISIBLE_DEVICES"]="0"
with tf.device('/GPU:0'):
    history=model.fit_generator(training_set,
                                steps_per_epoch=16,
                                epochs=100,
                                validation_data=test_gen,
                                validation_steps=28,
                                callbacks=callbacks_list)


x_test, y_test = [],[]
for ibatch, (x, y) in enumerate(test_gen):
    X_test.append(x)
    y_test.append (y)
    ibatch+=1
    if (batch == 5*28): break
# Concatenate everything together
x_test = np.concatenate(x_test)
y_test = np.concatenate(y_test)
y_test = np.int32([np. argmax (r) for r in y_test])
#E Get the predictions from the model and calculate the accuracy
y_pred = np.int32([np.argmax (r) for r in model.predict (x_test)])
match = (y_test == y_pred)
print('Testing Accuracy-%.2f%%' %(np.sum(match)*100/match.shape[0]))
