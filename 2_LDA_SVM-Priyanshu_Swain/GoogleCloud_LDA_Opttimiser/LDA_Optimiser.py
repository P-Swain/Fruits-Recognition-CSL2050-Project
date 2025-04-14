import numpy as np

import os
import tempfile

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Function to Load Feature Vectors and Labels
def loadFeaturesAndLabels(features_path):
    features, labels = [], []

    if not os.path.exists(features_path):
        raise ValueError(f"Path {features_path} does not exist.")
    if not os.path.isdir(features_path):
        raise ValueError(f"Path {features_path} is not a directory.")

    fruit_folders = [folder for folder in os.listdir(features_path) if os.path.isdir(os.path.join(features_path, folder))]

    for fruit_folder in tqdm(fruit_folders, unit="folder", desc=f"Loading Features from {features_path}"):
        fruit_folder_path = os.path.join(features_path, fruit_folder)
        npy_files = [f for f in os.listdir(fruit_folder_path) if f.endswith('.npy')]

        for featureVectorFile in npy_files:
            feature_file_path = os.path.join(fruit_folder_path, featureVectorFile)
            labels.append(fruit_folder[:-2].strip())
            features.append(np.load(feature_file_path))

    return np.array(features), np.array(labels)


def loadData():
    trainX, trainY = loadFeaturesAndLabels('img_ColourHist_LBP_Hist_Features/Training')
    testX, testY = loadFeaturesAndLabels('img_ColourHist_LBP_Hist_Features/Testing')
    scaler = StandardScaler()

    # normalizing
    scaledTrainX = scaler.fit_transform(trainX)
    scaledTestX = scaler.transform(testX)
    return scaledTrainX, trainY, scaledTestX, testY

def LDA_SVM(trainX, trainY, testX, testY, rangeLDA):

    noOfLabels = len(np.unique(trainY))
    accuracyList = []

    for n in tqdm(rangeLDA, desc="Training LDA+SVM Models", unit="model"):

        # performing LDA
        LDA = LinearDiscriminantAnalysis(n_components=n)
        LDA.fit(trainX, trainY)
        trainX_LDA = LDA.transform(trainX)
        testX_LDA = LDA.transform(testX)

        SVM = SVC(kernel='linear') # linear SVM
        SVM.fit(trainX_LDA, trainY) # training SVM

        predTest = SVM.predict(testX_LDA)
        acc = accuracy_score(testY, predTest)
        accuracyList.append(acc * 100)

    return accuracyList


def plotAccuracies(rangeLDA, accuracies, save_path='/home/user/'):

    save_path = os.path.join(save_path, 'LDA_SVM_Accuracies_Plot2.png') # defifnigng save path for plotted graph

    plt.figure(figsize=(10, 6))
    plt.plot(rangeLDA, accuracies, marker='x')

    plt.title('LDA + SVM Test Accuracy vs Number of LDA Components')
    plt.xlabel('Number of LDA Components')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)

    plt.savefig(save_path, dpi=400)
    print(f"Plot saved as {save_path}")

    plt.close()


def main():

    trainX, trainY, testX, testY = loadData() # loading data

    # printing data description
    print(f"Training Features Shape: {trainX.shape}")
    print(f"Testing Features Shape: {testX.shape}")
    print(f"Number of Labels: {len(np.unique(trainY))}")

    rangeLDA = list(range(31, 70, 3))  # 5 to 65
    accuracyList = LDA_SVM(trainX, trainY, testX, testY, rangeLDA)

    print("\nAccuracies for the LDA-SVM Models:")
    for n, acc in zip(rangeLDA, accuracyList):
        print(f"LDA Components {n:<4} Test Accuracy = {acc:.2f}%")

    plotAccuracies(rangeLDA, accuracyList)


if __name__ == "__main__":
    main()
