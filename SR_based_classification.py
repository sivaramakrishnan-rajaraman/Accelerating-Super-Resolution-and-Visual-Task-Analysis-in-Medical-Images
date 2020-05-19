#%% Super resolution based classification

#%% load libraries

import tensorflow as tf
tf.keras.backend.clear_session()

#%%
import time
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.utils import multi_gpu_model,np_utils
from keras.models import Model, model_from_json, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.losses import mean_absolute_error
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers import Conv2D, BatchNormalization, Input, add, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
import cv2
from PIL import Image
from sklearn.metrics import log_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, accuracy_score
from skimage.measure import compare_ssim
from matplotlib import pyplot as plt
from keras.preprocessing import image


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #if using multiple gpus, otherwise comment


#%%
##load libraries
#import cv2
#import numpy as np
#import os
#from keras.utils import np_utils
#import matplotlib.pyplot as plt
#import itertools
#import math
#from keras.models import Model
#from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
#from keras.layers import Input, Conv2D, Activation, Dense, BatchNormalization, SeparableConv2D, MaxPooling2D, Flatten, Dropout, UpSampling2D, GlobalAveragePooling2D, GaussianNoise, Reshape
#from keras.optimizers import SGD, RMSprop, Adam
#from sklearn.metrics import precision_recall_curve, average_precision_score, matthews_corrcoef, mean_squared_error,  mean_squared_log_error
#from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
#from tqdm import tqdm
#import keras
#from keras.models import Sequential, Model, Input, load_model
#from keras.layers import Conv2D, Dense, MaxPooling2D, SeparableConv2D, Activation, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D,Flatten,Average, BatchNormalization, Dropout
#import time
#import keras
#import cv2
#import imutils
#from keras_radam import RAdam
#from keras.preprocessing.image import img_to_array
#import pickle
#from keras.optimizers import Adam, SGD
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import average_precision_score
#from sklearn.metrics import matthews_corrcoef
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_squared_log_error
#from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, accuracy_score
#import matplotlib.pyplot as plt
#from keras.callbacks import ModelCheckpoint
#import scikitplot as skplt
#from itertools import cycle
#from sklearn.utils import class_weight
#from sklearn.preprocessing import LabelBinarizer
#import numpy as np
#import itertools
#from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.xception import Xception
#from keras.applications.resnet50 import ResNet50
#from keras.applications.densenet import DenseNet121
#from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import to_categorical
#from keras import backend as K
#from keras.models import model_from_json, load_model
#from keras.preprocessing import image
#import numpy as np
#import imageio
#import cv2
#from PIL import Image
#from networks.vdsr import build_model
#from sklearn.metrics import log_loss
#from sklearn.metrics import average_precision_score
#from sklearn.metrics import roc_curve, auc
#import time
#import numpy as np
#import math
#from skimage.measure import compare_ssim
#from matplotlib import pyplot as plt
#import argparse
#import imutils
#import cv2
#import tensorflow as tf
#from tensorflow.losses import huber_loss
#from keras.optimizers import SGD, Adam
#from networks.upsample import build_model
#from modules.file import save_model
#from modules.image import load_images, load_images_scale, to_dirname
#from modules.interface import show
#from keras import backend as K
#from keras.losses import mean_squared_error, mean_absolute_error, logcosh
#from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

#%%define data directories
train_data_dir = 'cxr_sr_classification/shenzhen/train' #path to your data
test_data_dir = 'cxr_sr_classification/shenzhen/test'

# declare the number of samples in each category
nb_train_samples = 464 
nb_test_samples = 198 
num_classes = 2
out_img_row = 256
out_img_col = 256
img_chan = 1 # modify for RGB images
batch_size = 8 #modify
epochs = 32 #modify

#%%
# define functions to load and resize the training, validation and test data.
def load_data(data_dir: str, img_num: int, img_chan: int,
              save_numpy_path: str="", save_as_numpy: bool=True,
              resize: bool = True, outsize: tuple =(256,256)): #modify to your requirements
    print("Loading images", end='')
    if os.path.isfile(save_numpy_path) and save_as_numpy is True:
        print(" from " + save_numpy_path)
        npzfile = np.load(save_numpy_path)
        X = npzfile['arr_0']
        Y = npzfile['arr_1']

    else:
        print(" from orginal images")
        labels = os.listdir(data_dir)
        num_classes = len(labels)
        X = np.ndarray((img_num, outsize[0], outsize[1], img_chan), dtype=np.float32)
        Y = np.zeros((img_num,), dtype=np.float32)
        i = 0
        print('-'*30)
        print('Creating dataset now...')
        print('-'*30)
        j = 0
        img_index = 0  # range from 0 to X's img num
        for label in labels:
            image_names_train = os.listdir(os.path.join(data_dir, label))
            total = len(image_names_train)
            print("Loading subfolder ", label, total, " images")
            for i in tqdm(range(len(image_names_train))):
                img = cv2.imread(os.path.join(data_dir, label, image_names_train[i]), cv2.IMREAD_COLOR)
                # preprocessing
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # comment if using rgb images
                if resize is True:
                    img = cv2.resize(img, outsize)
                    width = int(img.shape[1])
                    height = int(img.shape[0])
                    dim = (width, height)
                    dim1 = (width//2, height//2) #modify for different scales
                    img1 = cv2.resize(img, dim1)
                    img = cv2.resize(img1, dim, interpolation = cv2.INTER_CUBIC)
                img = img.astype('float32')
                img = img / 255
                img = np.expand_dims(img, axis=2) # modify for rgb images
                X[img_index] = img
                Y[img_index] = j
                img_index += 1
            j += 1
        print(i)
        print('Loading done.')
        """
        plt.hist(np.reshape(X, (-1)))
        plt.show()
        """
        print('Transform targets to keras compatible format.')
        Y = np_utils.to_categorical(Y[:img_num], num_classes)
        if save_as_numpy is True:
            if save_numpy_path == '':
                raise ValueError("The name of the save file is not assigned!")
            if save_numpy_path.split(".")[-1] != 'npz':
                raise ValueError("The name shoud have a npy extension")
            np.savez(save_numpy_path, X, Y)  # save as numpy files
        print("Conversion down, please use npy file later!")
    return X, Y

#%% load data
X_train, Y_train = load_data(train_data_dir, nb_train_samples,
                             img_chan, outsize=(out_img_row, out_img_col),
                             save_numpy_path="data/imgs_train_scale2_size_256.npz")
# do the test later
X_test, Y_test = load_data(test_data_dir, nb_test_samples, 
                           img_chan, outsize =(out_img_row, out_img_col),
                           save_numpy_path = "data/imgs_test_scale2_size_256.npz")

# print the shape of the data
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#%% custom lookahead function for model optimizer

class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (model.updates +
                                training_updates +
                                model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R
                model.train_function = F
                
#%% confusion matrix function
                
def plot_confusion_matrix(cm, classes,
                          normalize=False,  # if true all values in confusion matrix is between 0 and 1
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%% a baseline custom model and train on the interpolated LR images
    
input_img = Input(shape=(out_img_row, out_img_col, img_chan))
x1 = Conv2D(16, (3, 3), padding='same', 
            activation='relu', name ='conv1_new')(input_img)
x2 = MaxPooling2D(pool_size=(2, 2)) (x1)

x3 = Conv2D(32, (3, 3), padding='same', 
            activation='relu', name ='conv2_new')(x2)
x4 = MaxPooling2D(pool_size=(2, 2)) (x3)

x5 = Conv2D(64, (3, 3), padding='same',
            activation='relu', name ='conv3_new')(x4)
x6 = MaxPooling2D(pool_size=(2, 2)) (x5)

x7 = Conv2D(128, (3, 3), padding='same', 
            dilation_rate=2, activation='relu', name ='conv4_new')(x6)
x8 = MaxPooling2D(pool_size=(2, 2)) (x7)

x9 = Conv2D(256, (3, 3), padding='same', 
            dilation_rate=2, activation='relu', name ='conv5_new')(x8)
x10 = MaxPooling2D(pool_size=(2, 2)) (x9)

x11 = Conv2D(512, (3, 3), padding='same', 
             dilation_rate=2, activation='relu', name ='conv6_new')(x10) # use dilation in deeper layers to increase receptive field

flat = GlobalAveragePooling2D()(x11)
predictions = Dense(num_classes, activation='softmax', name ='dense_new')(flat)

model_base = Model(inputs=input_img, outputs=predictions, 
                  name = 'baseline_classification_model')
# summarize layers
print(model_base.summary())   

#enumerate and print layer names
for i, layer in enumerate(model_base.layers):
   print(i, layer.name)   

#%% fix the optimizer compile the custom model
   
model_base.compile(optimizer=Adam(lr=0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
lookahead.inject(model_base) # add into model

#%% start training

filepath1 = 'weights/classification/baseline/' + model_base.name + '.best.h5'
checkpoint1 = ModelCheckpoint(filepath1, monitor='val_loss', verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, mode='min', period=1)
earlyStopping1 = EarlyStopping(monitor='val_loss', 
                               patience=15, verbose=1, mode='min')
tensor_board1 = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                              verbose=1, mode='min', min_lr=0.00001) #modify if using validation accuracy and mode=max
callbacks_list = [checkpoint1, tensor_board1, earlyStopping1, reduce_lr1]

#begin training
t=time.time() 
print('-'*30)
print('Start Training the model...')
print('-'*30)
history = model_base.fit(X_train, Y_train, batch_size=batch_size,
                          validation_data=(X_test, Y_test),
                          callbacks=callbacks_list,
                          shuffle=True,
                          epochs=epochs, verbose=1)
print('Training time: %s' % (time.time()-t))

#print the history of the trained model
print(history.history)

#%% # plot the loss plot between training and validation data to visualize the model performance.

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
loss_np = np.array(loss)
val_loss_np = np.array(val_loss)
np.savez("loss", loss_np, val_loss_np)


#%% load the saved weights

model = load_model('weights/classification/baseline/baseline_classification_model.best.h5')
model.summary()

#fix optimizer
model.compile(optimizer=Adam(lr=0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
lookahead.inject(model) # add into model

# Make predictions
print('-'*30)
print('Predicting on test data...')
print('-'*30)
y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

# computhe the cross-entropy loss score
score = log_loss(Y_test,y_pred)
print(score)

# compute the average precision score
prec_score = average_precision_score(Y_test,y_pred)  
print(prec_score)

# compute the accuracy on test data
Test_accuracy = accuracy_score(Y_test.argmax(axis=-1),y_pred.argmax(axis=-1))
print("Test_Accuracy = ",Test_accuracy)

#%%
#compute the ROC-AUC values
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fig=plt.figure(figsize=(15,10), dpi=70)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

lw = 1 #true class label
plt.plot(fpr[1], tpr[1], color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right")
plt.show()

#%%
#declare target names
target_names = ['abnormal', 'normal'] #modify

#print classification report
print(classification_report(Y_test.argmax(axis=-1),y_pred.argmax(axis=-1),target_names=target_names))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test.argmax(axis=-1),y_pred.argmax(axis=-1))
np.set_printoptions(precision=4)

# Plot non-normalized confusion matrix
plt.figure(figsize=(20,10), dpi=70)
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')

plt.show()

#%% save predictions

y_pred = np.argmax(y_pred, axis=1)
Y_test = np.argmax(Y_test, axis=1)

print(y_pred)
print(Y_test)

np.savetxt('y_pred.csv',y_pred,fmt='%i',delimiter = ",")
np.savetxt('Y_test.csv',Y_test,fmt='%i',delimiter = ",")

#%% examine the performance with the freezed SR weights and add the classification model to the truncated deepst convolutional layer

json_file = open('weights/scale_2/model.json', 'r') #modify depending on the LR images you use, i.e. scale 2 model for scale 2 LR images
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights/VDSRCNN.best.hdf5")
print("Loaded model from disk")
loaded_model.summary()

#enumerate and print layer names
for i, layer in enumerate(loaded_model.layers):
   print(i, layer.name)

#%% #truncate at the deepest convolutional layer
model_new = Model(inputs=loaded_model.input, 
                    outputs=loaded_model.get_layer('conv2d_19').output) 
x = model_new.output

#%% add the baseline classification model

x1 = Conv2D(16, (3, 3), padding='same', 
            activation='relu', name ='conv1_new')(x)
x2 = MaxPooling2D(pool_size=(2, 2)) (x1)
x3 = Conv2D(32, (3, 3), padding='same', 
            activation='relu', name ='conv2_new')(x2)
x4 = MaxPooling2D(pool_size=(2, 2)) (x3)
x5 = Conv2D(64, (3, 3), padding='same', 
            activation='relu', name ='conv3_new')(x4)
x6 = MaxPooling2D(pool_size=(2, 2)) (x5)
x7 = Conv2D(128, (3, 3), padding='same', 
            dilation_rate=2, activation='relu', name ='conv4_new')(x6)
x8 = MaxPooling2D(pool_size=(2, 2)) (x7)
x9 = Conv2D(256, (3, 3), padding='same', 
            dilation_rate=2, activation='relu', name ='conv5_new')(x8)
x10 = MaxPooling2D(pool_size=(2, 2)) (x9)
x11 = Conv2D(512, (3, 3), padding='same', 
             dilation_rate=2, activation='relu', name ='conv6_new')(x10)
flat = GlobalAveragePooling2D()(x11)
predictions = Dense(num_classes, activation='softmax', name ='dense_new')(flat)
model_srcxr = Model(inputs=model_new.input, outputs=predictions, 
                  name = 'cxr_sr_classification_model')
# summarize layers
print(model_srcxr.summary())  

#enumerate and print layer names
for i, layer in enumerate(model_srcxr.layers):
   print(i, layer.name)
   
#%% freeze the super resolution weights and train only the newly added classification layers
   
for layers in  model_srcxr.layers[:38]: 
    layers.trainable = False
for layers in  model_srcxr.layers[38:]: #modify dependikng on the model used
    layers.trainable = True    

#%% #fix the optimizer compile the model
    
model_srcxr.compile(optimizer=Adam(lr=0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
lookahead.inject(model_srcxr) # add into model

#%% #start training

filepath1 = 'weights/classification/pretrained/vdsr/scale2/' + model_srcxr.name + '.best.h5'
checkpoint1 = ModelCheckpoint(filepath1, monitor='val_loss', verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, mode='min', period=1)
earlyStopping1 = EarlyStopping(monitor='val_loss', 
                               patience=15, verbose=1, mode='min')
tensor_board1 = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
reduce_lr1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                              verbose=1, mode='min', min_lr=0.00001)
callbacks_list = [checkpoint1, tensor_board1, earlyStopping1, reduce_lr1]

t=time.time() #make a note of the time
#start training
print('-'*30)
print('Start Training the model...')
print('-'*30)

history = model_srcxr.fit(X_train, Y_train, batch_size=batch_size,
                          validation_data=(X_test, Y_test),
                          callbacks=callbacks_list,
                          shuffle=True,
                          epochs=epochs, verbose=1)
print('Training time: %s' % (time.time()-t))

#print the history of the trained model
print(history.history)

#%% # plot the loss plot between training and validation data to visualize the model performance.
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
loss_np = np.array(loss)
val_loss_np = np.array(val_loss)
np.savez("loss", loss_np, val_loss_np)

#%% # load the saved model and predict

model = load_model('weights/classification/pretrained/vdsr/scale2/cxr_sr_classification_model.best.h5')
model.summary()

#fix optimizer
model.compile(optimizer=Adam(lr=0.001),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
lookahead.inject(model) # add into model

# Make predictions
print('-'*30)
print('Predicting on test data...')
print('-'*30)
y_pred_sr = model.predict(X_test, batch_size=batch_size, verbose=1)

#%% meausre performance metrics 

# compute the cross-entropy loss score
score = log_loss(Y_test,y_pred_sr)
print(score)

# compute the average precision score
prec_score = average_precision_score(Y_test,y_pred_sr)  
print(prec_score)

# compute the accuracy on test data
Test_accuracy = accuracy_score(Y_test.argmax(axis=-1),y_pred_sr.argmax(axis=-1))
print("Test_Accuracy = ",Test_accuracy)

#%% #compute the ROC-AUC values

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred_sr[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fig=plt.figure(figsize=(15,10), dpi=70)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
major_ticks = np.arange(0.0, 1.0, 0.05)
minor_ticks = np.arange(0.0, 1.0, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')

lw = 1 
plt.plot(fpr[1], tpr[1], color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right")
plt.show()

#%% #declare target names

target_names = ['abnormal', 'normal'] #modify

#print classification report
print(classification_report(Y_test.argmax(axis=-1),y_pred_sr.argmax(axis=-1),target_names=target_names))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test.argmax(axis=-1),y_pred_sr.argmax(axis=-1))
np.set_printoptions(precision=4)

# Plot non-normalized confusion matrix
plt.figure(figsize=(20,10), dpi=70)
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')

plt.show()

#%% store the predictions

y_pred_sr1 = np.argmax(y_pred_sr, axis=1)
Y_test1 = np.argmax(Y_test, axis=1)

print(y_pred_sr1)
print(Y_test1)

np.savetxt('y_pred_sr.csv',y_pred_sr1,fmt='%i',delimiter = ",")
np.savetxt('Y_test_sr.csv',Y_test1,fmt='%i',delimiter = ",")

#%%
''' the performance with SR freezed weights and training the classification model
was above par compared to training a baseline custom model with random initialized weights
'''

#%%