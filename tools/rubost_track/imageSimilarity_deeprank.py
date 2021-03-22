# coding: utf-8

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Embedding
from tensorflow.keras import backend as K

import cv2

class imageSimilarity_deeprank():
    def __init__(self):
        self._model_path = '/home/rislab/Workspace/Image-Similarity-Deep-Ranking/deepranking-v2-150000.h5'

        self.model = self.deep_rank_model()

        # for layer in model.layers:
        #     print (layer.name, layer.output_shape)

        self.model.load_weights(self._model_path)

    def convnet_model_(self):
        vgg_model = VGG16(weights=None, include_top=False)
        x = vgg_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Lambda(lambda x_: K.l2_normalize(x, axis=1))(x)
        convnet_model = Model(inputs=vgg_model.input, outputs=x)
        return convnet_model

    def convnet_model(self):
        vgg_model = VGG16(weights=None, include_top=False)
        x = vgg_model.output
        x - GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)

    def deep_rank_model(self):
        convnet_model = self.convnet_model_()
        first_input = Input(shape=(224, 224, 3))
        first_conv = Conv2D(96, kernel_size=(8, 8), strides=(16, 16), padding='same')(first_input)
        first_max = MaxPool2D(pool_size=(3, 3), strides=(4, 4), padding='same')(first_conv)
        first_max = Flatten()(first_max)
        first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

        second_input = Input(shape=(224, 224, 3))
        second_conv = Conv2D(96, kernel_size=(8, 8), strides=(32, 32), padding='same')(second_input)
        second_max = MaxPool2D(pool_size=(7, 7), strides=(2, 2), padding='same')(second_conv)
        second_max = Flatten()(second_max)
        second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)

        merge_one = concatenate([first_max, second_max])

        merge_two = concatenate([merge_one, convnet_model.output])
        emb = Dense(4096)(merge_two)
        l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

        final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

        return final_model

    def preprocess_cvimage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        return x

    def get_feature(self,cvimg):
        img = self.preprocess_cvimage(cvimg)
        embedding = self.model.predict([img, img, img])[0]
        return embedding

    def similarity_score(self,feature1,feature2):
        distance=np.sqrt(((feature1 - feature2) ** 2).sum())
        return distance

    def run(self, cvimg1, cvimg2):

        feature1 = self.get_feature(cvimg1)
        feature2 = self.get_feature(cvimg2)

        distance2=self.similarity_score(feature1,feature2)

        distance = sum([(feature1[idx] - feature2[idx]) ** 2 for idx in range(len(feature1))]) ** (0.5)

        print('dist1= %f, dist2 = %f' %(distance,distance2))
        return distance

if __name__ == '__main__':
    sim=imageSimilarity_deeprank()

    img1 = cv2.imread('/home/rislab/Workspace/pysot/testing_dataset/VOT2018/crossing/00000001.jpg')
    img2 = cv2.imread('/home/rislab/Workspace/pysot/testing_dataset/VOT2018/dinosaur/00000130.jpg')
    dist = sim.run(img1,img2)

    print(dist)