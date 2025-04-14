import gradio as gr

import numpy as np
import cv2
from skimage.feature import local_binary_pattern as lbp

import joblib

lda_model_filename = 'LDA_Model_3.pkl'
LDA_Model = joblib.load(lda_model_filename)

lda_svm_model_filename = 'LDA_SVM_Model_3.pkl'
LDA_SVM_Model = joblib.load(lda_svm_model_filename)

scaler = joblib.load('LDA_SVM_Scaler.pkl')

def convertImgToNumpyArr(image):
    img = image.resize((100, 100))
    img = np.array(img)
    return img

def computeLBP_Features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbpImg = lbp(gray, method="uniform", P = 8, R = 1)
    bins = np.arange(0, 11)
    lbpHist, _ = np.histogram(lbpImg.flatten(), bins=bins, range=(0, 10))
    return lbpHist / (np.sum(lbpHist) + 1e-6)

def computeColourHist(image, bins=20):
    hsvImg = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsvImg = hsvImg / 255.0
    hueHist = np.histogram(hsvImg[:,:,0], bins=bins, range=(0, 1))[0]
    satHist = np.histogram(hsvImg[:,:,1], bins=bins, range=(0, 1))[0]
    valHist = np.histogram(hsvImg[:,:,2], bins=bins, range=(0, 1))[0]
    return np.concatenate((hueHist, satHist, valHist))

def predict(img):
    img_as_numpy_arr = convertImgToNumpyArr(img)
    featureMap = {}

    featuresDescriptors = ['colour', 'texture']

    if('colour' in featuresDescriptors):
        colourHistFeatures = computeColourHist(img_as_numpy_arr)
        featureMap['colour'] = colourHistFeatures

    if('texture' in featuresDescriptors):
        lbpFeatures = computeLBP_Features(img_as_numpy_arr)
        featureMap['texture'] = lbpFeatures

    featureVector = np.concatenate([featureMap[feature.lower()] for feature in featuresDescriptors if feature.lower() in featureMap])
    featureVector = featureVector.astype(np.float32)

    testFeatures = scaler.transform([featureVector])

    testFeaturesLDA = LDA_Model.transform(testFeatures)

    testPredictions_LDA_SVM = LDA_SVM_Model.predict(testFeaturesLDA)

    return str(testPredictions_LDA_SVM[0])

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Classification with LDA + SVM",
    description="Upload a JPG image to get the predicted class."
)

interface.launch()
