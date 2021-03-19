'''Image similarity using deep features.

Recommendation: the threshold of the `DeepModel.cosine_distance` can be set as the following values.
    0.84 = greater matches amount
    0.845 = balance, default
    0.85 = better accuracy
'''

from io import BytesIO
import os
import numpy as np

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import cv2

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.preprocessing import image as process_image
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras import Model

class ImageSimilarity():
    '''Image similarity.'''
    def __init__(self):
        self._model = self._define_model()

    @staticmethod
    def _define_model(output_layer=-1):
        '''Define a pre-trained MobileNet model.

        Args:
            output_layer: the number of layer that output.

        Returns:
            Class of keras model with weights.
        '''
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = base_model.layers[output_layer].output
        output = GlobalAveragePooling2D()(output)
        model = Model(inputs=base_model.input, outputs=output)
        return model

    @staticmethod
    def cosine_distance(input1, input2):
        '''Calculating the distance of two inputs.

        The return values lies in [-1, 1]. `-1` denotes two features are the most unlike,
        `1` denotes they are the most similar.

        Args:
            input1, input2: two input numpy arrays.

        Returns:
            Element-wise cosine distances of two inputs.
        '''
        # return np.dot(input1, input2) / (np.linalg.norm(input1) * np.linalg.norm(input2))
        return np.dot(input1, input2.T) / \
                np.dot(np.linalg.norm(input1, axis=1, keepdims=True), \
                        np.linalg.norm(input2.T, axis=0, keepdims=True))


    def preprocess_image(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(224, 224))
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        return x

    def extract_feature(self,img):

        features = self._model.predict(img)
        return features

    def similarity_score(self,feature1,feature2):
        thresh = 0.845

        distances = self.cosine_distance(feature1, feature2)

        return distances

    def get_feature(self,cv_img):
        img=self.preprocess_image(cv_img)
        feature = self.extract_feature(img)
        return feature

    def cal_similarity_score(self,cv_img1,cv_img2):
        img1=self.preprocess_image(cv_img1)
        img2=self.preprocess_image(cv_img2)

        features1=self.extract_feature(img1)
        features2 = self.extract_feature(img2)

        dist = self.similarity_score(features1,features2)

        return dist


if __name__ == '__main__':
    similarity = ImageSimilarity()

    img1=cv2.imread('/home/rislab/Workspace/image-similarity/demo/1.jpg')
    img2=cv2.imread('/home/rislab/Workspace/image-similarity/demo/2.jpg')

    dist=similarity.cal_similarity_score(img1,img2)

    print('score: %f' %(dist))
