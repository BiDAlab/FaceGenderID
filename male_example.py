import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
import cv2
import numpy as np
from scipy import misc
from keras_vggface import utils
from keras.preprocessing import image
from sklearn import preprocessing
from scipy.spatial import distance
import os

def l2_norm(x):
    #x2=tf.nn.l2_normalize(x,axis=None,epsilon=1e-12,name=None,dim=None)
    x2=tf.nn.l2_normalize(x,dim=1)
    return x2


# CHOOSE AN ARCHITECTURE:  
architecture = 'vgg'
#architecture = 'resnet'


dirpath = os.getcwd()
img_url_1 = dirpath + '/images/n005123/0015_01.jpg'
img_url_2 = dirpath + '/images/n005123/0077_01.jpg'
#img_url_2 = dirpath + '/images/n005123/0041_01.jpg'

bbox_1 = [65,66,125,166]
bbox_2 = [118,103,194,260]
#bbox_2 = [90,50,142,182] 


if architecture == 'vgg':
	#baseline model with vgg:
	vgg_model = VGGFace(model = 'vgg16')
	feature_layer = vgg_model.get_layer('fc6/relu').output
	model = Model(vgg_model.input, feature_layer)
	version = 1 
	#proposed model with vgg (trained with triplet loss)::
	proposed_model = Sequential()
	proposed_model.add(Dense(1024, input_dim=4096, activation='linear')) #vgg  
	proposed_model.add(Dropout(0.1))  
	proposed_model.add(Lambda(l2_norm))
	proposed_model.load_weights(dirpath + '/weights/male_vgg.h5') #load appropriate weights
    
elif architecture == 'resnet':
	#baseline model with resnet:
    resnet = VGGFace(model = 'resnet50')
    last_layer = resnet.get_layer('avg_pool').output
    feature_layer = Flatten(name='flatten')(last_layer)
    model = Model(resnet.input, feature_layer)
    version = 2   
    #proposed model with resnet (trained with triplet loss):
    proposed_model = Sequential()
    proposed_model.add(Dense(1024, input_dim=2048, activation='linear')) #resnet     
    proposed_model.add(Dropout(0.1))  
    proposed_model.add(Lambda(l2_norm))
    proposed_model.load_weights(dirpath + '/weights/male_resnet.h5') #load appropriate weights
	

# first image:
img_1 = cv2.imread(img_url_1)
img_1 = img_1/255
img_1 = img_1[bbox_1[1]:bbox_1[1]+bbox_1[3],bbox_1[0]:bbox_1[0]+bbox_1[2],:]
img_1 = misc.imresize(img_1, (224, 224), interp='bilinear')
img_1 = image.img_to_array(img_1)
img_1 = np.expand_dims(img_1, axis=0)
img_1 = utils.preprocess_input(img_1, version=version) # version=1 for vgg16, version=2 for resnet50
predict_1 = model.predict(img_1) # predict with baseline model
baseline_features_1 = preprocessing.normalize(predict_1, norm='l2', axis=1, copy=True, return_norm=False)
proposed_features_1 = proposed_model.predict(predict_1) # predict with proposed model

# second image:
img_2 = cv2.imread(img_url_2)
img_2 = img_2/255
img_2 = img_2[bbox_2[1]:bbox_2[1]+bbox_2[3],bbox_2[0]:bbox_2[0]+bbox_2[2],:]
img_2 = misc.imresize(img_2, (224, 224), interp='bilinear')
img_2 = image.img_to_array(img_2)
img_2 = np.expand_dims(img_2, axis=0)
img_2 = utils.preprocess_input(img_2, version=version) # version=1 for vgg16, version=2 for resnet50
predict_2 = model.predict(img_2) # predict with baseline model
baseline_features_2 = preprocessing.normalize(predict_2, norm='l2', axis=1, copy=True, return_norm=False)
proposed_features_2 = proposed_model.predict(predict_2) # predict with proposed model

# euclidean distances:
baseline_distance = distance.euclidean(baseline_features_1, baseline_features_2)
proposed_distance = distance.euclidean(proposed_features_1, proposed_features_2)




