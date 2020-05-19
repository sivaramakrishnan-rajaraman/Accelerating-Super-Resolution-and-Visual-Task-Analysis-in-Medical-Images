#segmentation using super resolution weights as initialization

#%% load libraries
import tensorflow as tf
tf.keras.backend.clear_session()

#%%
import time
import cv2
import numpy as np
import tensorflow as tf
#from keras.utils import multi_gpu_model
from keras.models import Model, model_from_json, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D
import skimage.transform as trans
import skimage.io as io
from keras.preprocessing import image
import imutils
from keras.preprocessing.image import img_to_array
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #if using multiple gpus, otherwise comment

#%% #define loss and performance metrics

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    
def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y

def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

#%% Construct a sequential fully convolutional network with encoder and symmetrical decoder
    
fcn = Sequential()
fcn.add(Conv2D(8, kernel_size=3, padding='same', activation='relu', 
               input_shape = (256,256,1), name = 'enc_conv1')) #modify the channels to 1 if grayscale
fcn.add(Conv2D(16, kernel_size=3, padding='same', 
               activation='relu', name = 'enc_conv2'))
fcn.add(Conv2D(32, kernel_size=3, padding='same', 
               activation='relu', name = 'enc_conv3'))
fcn.add(Conv2D(64, kernel_size=3, padding='same', 
               dilation_rate=2, activation='relu', name = 'enc_conv4')) #add dilation layers to increase the receptive field in deeper layers to help semantic segmentation
fcn.add(Conv2D(128, kernel_size=3, padding='same', 
               dilation_rate=2, activation='relu', name = 'enc_conv5'))
fcn.add(Conv2D(256, kernel_size=3, padding='same', 
               dilation_rate=2, activation='relu', name = 'enc_conv6'))
fcn.add(Conv2D(128, kernel_size = 3, 
               padding = 'same', name = 'dec_conv5'))
fcn.add(Conv2D(64, kernel_size = 3, 
               padding = 'same', name = 'dec_conv4'))
fcn.add(Conv2D(32, kernel_size = 3, 
               padding = 'same', name = 'dec_conv3'))
fcn.add(Conv2D(16, kernel_size = 3, 
               padding = 'same', name = 'dec_conv2'))
fcn.add(Conv2D(8, kernel_size = 3, 
               padding = 'same', name = 'dec_conv1'))
fcn.add(Conv2D(1, kernel_size = 1, 
               padding = 'same', activation = 'sigmoid'))
fcn.summary() 
#compile the model and train from the scratch
fcn.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy',iou,iou_thresholded]) #try bce_dice_loss as well and record the best

#%% Define the data format and the functions to train and test with the data using image generators
#declare values for the color dictionary
v1 = [128,128,128]
v2 = [128,0,0]
v3 = [192,192,128]
v4 = [128,64,128]
v5 = [60,40,222]
v6 = [128,128,0]
v7 = [192,128,128]
v8 = [64,64,128]
v9 = [64,0,128]
v10 = [64,64,0]
v11 = [0,128,192]
v12 = [0,0,0]

COLOR_DICT = np.array([v1, v2, v3, v4, v5, v6, 
                       v7, v8, v9, v10, v11, v12])

#reshape mask dimensions
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],
                                        new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

#generator for training images
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale", #modify to rgb if using color images for images and mask
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256, 256),seed = 1): 

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
        
def valGenerator(batch_size,val_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale", #modify to rgb if using color images for images and mask
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256, 256),seed = 1): 

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    val_generator = zip(image_generator, mask_generator)
    for (img,mask) in val_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def testGenerator(test_path,target_size = (256, 256),flag_multi_class = False, as_gray = True): #make as_gray = False for RGB images
    for filename in os.listdir(test_path):
        img = io.imread(os.path.join(test_path,filename),as_gray = as_gray) 
        img = img / 255.
        img = trans.resize(img,target_size) # comment for rgb images
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        #img = np.reshape(img,img.shape+(3,)) if (not flag_multi_class) else img #for RGB images
        img = np.reshape(img,(1,)+img.shape)
        yield img


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult(save_path,npyfile,test_path, flag_multi_class = False, num_class = 2): 
    file_names = os.listdir(test_path)
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,file_names[i]),img)
            
#%% declare the test and result path
        
test_path = "cxr_sr_segmentation_scale3/test"
save_path = "cxr_sr_segmentation_scale3/result/predicted_mask"

#give the data augmentation parameters to prevent model overfitting
data_gen_args = dict(rotation_range=5.,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    shear_range=0.0,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')  

myGene = trainGenerator(4,'cxr_sr_segmentation_scale3/train',
                        'image','label',data_gen_args,save_to_dir = None) 
valGene = valGenerator(4,'cxr_sr_segmentation_scale3/val',
                       'image','label',data_gen_args,save_to_dir = None) #change batch size if having better gpus

callbacks = [EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-4,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                               epsilon=1e-4, mode='min'),
             ModelCheckpoint(monitor='val_loss', 
                             filepath='cxr_sr_segmentation_scale3/result/trained_model/fcn.hdf5', 
                             save_best_only=True,
                             mode='min', verbose = 1)]
t=time.time() 
print('-'*30)
print('Start Training the Segmentation model...')
print('-'*30)

#start model training
#steps_per_epoch= number of training samples//batch_size; validation_steps = number of validation samples//batch_size
fcn.fit_generator(generator=myGene,steps_per_epoch=137, epochs=200, callbacks=callbacks, 
                    validation_data=valGene, validation_steps=35, verbose=1) 

print('Training time: %s' % (time.time()-t))

#%% predict on the test data

testGene = testGenerator(test_path, flag_multi_class=False, target_size=(256,256), as_gray=True) # make as_gray=False and flag_multi_class = True for RGB images
fcn.load_weights('cxr_sr_segmentation_scale3/result/trained_model/fcn.hdf5')
results = fcn.predict_generator(testGene,138,verbose=1, workers=1, use_multiprocessing=False) #modify according to number of test samples, here it is 138
saveResult(save_path, results, test_path)  

#%% Predict on a single image

img = cv2.imread('cxr_sr_segmentation_scale3/test/image_1.png')
output = imutils.resize(img, width=400)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale
img = cv2.resize(image, (256, 256))
width = int(img.shape[1])
height = int(img.shape[0])
dim = (width, height)
dim1 = (width//2, height//2) #modify for different scales
img1 = cv2.resize(img, dim1)
# resize image
image = cv2.resize(img1, dim, interpolation = cv2.INTER_CUBIC)                
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
fcn.load_weights('cxr_sr_segmentation_scale3/result/trained_model/fcn.hdf5')

#predict time
t=time.time() 
print('-'*30)
print('performing segmentation...')
print('-'*30)
results = fcn.predict(image,1,verbose=1) 
print('Segmentation time: %s' % (time.time()-t))

#%% Lets see the impact of loading the super resolution model weights
#to initialize and train the segmentation model

#load the super resolution model VDSR weights

json_file = open('weights/model.json', 'r') #modify the path depending on the scale factor. i.e. use the model validated with scale 3 weights if using scale 3 images
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("weights/VDSRCNN.best.hdf5")
print("Loaded model from disk")
loaded_model.summary()

#enumerate and print layer names
for i, layer in enumerate(loaded_model.layers):
   print(i, layer.name)

#%% truncate the model at the deepest convolutional layer
   
model_new = Model(inputs=loaded_model.input, 
                    outputs=loaded_model.get_layer('conv2d_19').output) #modify depending on the model
x = model_new.output

#%% add the fully convolutional netework layers to the truncated super resolution model

x1 = Conv2D(8, kernel_size=3, padding='same', 
            activation='relu', name = 'enc_conv1')(x)
x2 = Conv2D(16, kernel_size=3, padding='same', 
            activation='relu', name = 'enc_conv2')(x1)
x3 = Conv2D(32, kernel_size=3, padding='same', 
            activation='relu', name = 'enc_conv3')(x2)
x4 = Conv2D(64, kernel_size=3, padding='same', 
            activation='relu', dilation_rate=2, name = 'enc_conv4')(x3)
x5 = Conv2D(128, kernel_size=3, padding='same', 
            activation='relu', dilation_rate=2, name = 'enc_conv5')(x4)
x6 = Conv2D(256, kernel_size=3, padding='same', 
            activation='relu', dilation_rate=2, name = 'enc_conv6')(x5)
x7 = Conv2D(128, kernel_size=3, padding='same', 
            activation='relu', name = 'dec_conv5')(x6)
x8 = Conv2D(64, kernel_size=3, padding='same', 
            activation='relu', name = 'dec_conv4')(x7)
x9 = Conv2D(32, kernel_size=3, padding='same', 
            activation='relu', name = 'dec_conv3')(x8)
x10 = Conv2D(16, kernel_size=3, padding='same', 
             activation='relu', name = 'dec_conv2')(x9)
x11 = Conv2D(8, kernel_size=3, padding='same', 
             activation='relu', name = 'dec_conv1')(x10)
predictions = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid', name = 'dec_conv0')(x11) #modify the channel for rgb images

fcn_sr = Model(inputs=model_new.input, outputs=predictions, 
                  name = 'fcn_sr_model')
# summarize layers
print(fcn_sr.summary())   

#enumerate and print layer names
for i, layer in enumerate(fcn_sr.layers):
   print(i, layer.name)   

#%% #freezing the super resolution weights didnt help me; only complete retrainig gave best performance
##super resolutional weights freeze
#for layers in  fcn_sr.layers[0:38]: 
#    layers.trainable = False
#for layers in  fcn_sr.layers[38:]: 
#    layers.trainable = True  

#%%  
#compile the model and train from the scratch
fcn_sr.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy',iou,iou_thresholded])

#%% declare the test and result path
test_path = "cxr_sr_segmentation_scale3/test"
save_path = "cxr_sr_segmentation_scale3/result/predicted_mask_sr"

#give the data augmentation parameters to prevent model overfitting
data_gen_args = dict(rotation_range=5.,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    shear_range=0.0,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')  

myGene = trainGenerator(4,'cxr_sr_segmentation_scale3/train',
                        'image','label',data_gen_args,save_to_dir = None) 
valGene = valGenerator(4,'cxr_sr_segmentation_scale3/val',
                       'image','label',data_gen_args,save_to_dir = None) 

callbacks = [EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-4,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                               epsilon=1e-4, mode='min'),
             ModelCheckpoint(monitor='val_loss', 
                             filepath='cxr_sr_segmentation_scale3/result/trained_model/fcn_sr.hdf5', 
                             save_best_only=True,
                             mode='min', verbose = 1)]
t=time.time() 
print('-'*30)
print('Start Training the Segmentation model...')
print('-'*30)

#train
fcn_sr.fit_generator(generator=myGene,steps_per_epoch=137, epochs=200, callbacks=callbacks,
                    validation_data=valGene, validation_steps=35, verbose=1) 

print('Training time: %s' % (time.time()-t))

#%% predict on the test data

testGene = testGenerator(test_path, flag_multi_class=False, target_size=(256,256), as_gray=True) #modify for rgb as before
fcn_sr.load_weights('cxr_sr_segmentation_scale3/result/trained_model/fcn_sr.hdf5')
results = fcn_sr.predict_generator(testGene,138,verbose=1, workers=1, use_multiprocessing=False) 
saveResult(save_path, results, test_path)  

#%%predicting a single image

img = cv2.imread('cxr_sr_segmentation_scale3/test/image_1.png')
output = imutils.resize(img, width=400)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(image, (256, 256))
width = int(img.shape[1])
height = int(img.shape[0])
dim = (width, height)
dim1 = (width//2, height//2) #modify for different scales
img1 = cv2.resize(img, dim1)
image = cv2.resize(img1, dim, interpolation = cv2.INTER_CUBIC)                
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
fcn_sr.load_weights('cxr_sr_segmentation_scale3/result/trained_model/fcn_sr.hdf5')
t=time.time() 
print('-'*30)
print('performing segmentation...')
print('-'*30)
results = fcn_sr.predict(image,1,verbose=1) 
print('Segmentation time: %s' % (time.time()-t))

#%%
'''The results show that the FCN model initialized with the super resolution weights
and retrained from the scratch resulted in a superior performance compared to other methods.
'''
#%%