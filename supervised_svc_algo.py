import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from PIL import Image


# path to training & testing dataset
train_path = "dataset/training"
train_labels = os.listdir(train_path)


# image datasets variables
image_size = (28, 28)
actualimg_h, newimg_h = 28, 28
colorscale = 'rgb'  # grayscale
colorval = 3


# variables to hold features and labels
X = []
Y = []


# convert images into numeric lists
# loop over all the labels in the folder
count = 1
for i, label in enumerate(train_labels):
    cur_path = train_path + "/" + label
    count = 1
    for image_path in glob.glob(cur_path + "/*.jpg"):
        # raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
        img = image.load_img(
            image_path, target_size=image_size, color_mode=colorscale)
        img = image.img_to_array(img)
        img = img.flatten()
        X.append(img)
        Y.append(label)
        count += 1

# convert to np array
X = np.asarray(X)
Y = np.asarray(Y)

print('Data shape: ', X.shape)
print('Data label shape: ', Y.shape)


# split the data into a training and testing set,
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=0)


print('Data shape: ', X_Train.shape)
print('Data label shape: ', Y_Train.shape)


model = SVC(kernel='poly', C=0.1, max_iter=500000, gamma='auto')

model.fit(X_Train, Y_Train)

Y_Pred = model.predict(X_Test)


# finally, we can use the accuracy_score utility to see the fraction of
# predicted labels that match their true value:
acc_score = accuracy_score(Y_Test, Y_Pred)
print('accuracy score: ', acc_score)
confusion_matrix(Y_Test, Y_Pred)


# print(model)
print('Training Score: {0}'.format(model.score(X_Train, Y_Train)))
print('Testing Score: {0}'.format(model.score(X_Test, Y_Test)))


# let's plot the result
test_images = X_Test.reshape(-1, newimg_h, newimg_h, colorval)

fig, axes = plt.subplots(6, 6, figsize=(56, 56), subplot_kw={
                         'xticks': [], 'yticks': []}, gridspec_kw=dict(hspace=1.1, wspace=0.3))

for i, ax in enumerate(axes.flat):
    img = Image.fromarray(test_images[i].astype('uint8'))
    img_as_img = img.convert("RGB")

    ax.imshow(img_as_img, cmap='binary', interpolation='nearest')
    ax.text(-.1, -0.31, str(Y_Test[i]), transform=ax.transAxes, color='black')
    ax.text(-.1, -0.6, str(Y_Pred[i]), transform=ax.transAxes,
            color='green' if (Y_Pred[i] == Y_Test[i]) else 'red')

plt.show()
