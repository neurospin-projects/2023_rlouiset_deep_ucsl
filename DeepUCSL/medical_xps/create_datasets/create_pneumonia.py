import os

import cv2
import numpy as npl
import pandas as pd
import torchvision

input_dir = "/home/robin/Documents/datasets/pneumonia/"

# To create the "_disease" version of the datasets, please just comment the lines that
# concatenate the healthy labels and/or images to X_*, y_*, and y_*_subtype

img_dims = 224

to_pil = torchvision.transforms.ToPILImage()

# test dataset construction
X_test = np.zeros((0, img_dims, img_dims))
y_test = []
y_test_subtype = []
for cond in ['/NORMAL/', '/PNEUMONIA/']:
    c = 0
    for img_name in (os.listdir(input_dir + 'test' + cond)):
        c = c + 1
        img = cv2.imread(input_dir + 'test' + cond + img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(img_dims, img_dims), interpolation=cv2.INTER_AREA)
        img = np.asarray(img)
        img = img / 255.

        if cond == '/NORMAL/':
            label = 0
            subtype = -1
        elif cond == '/PNEUMONIA/':
            if "bacteria" in img_name:
                label = 1
                subtype = 0
            elif "virus" in img_name:
                label = 1
                subtype = 1
            else:
                print(img_name)
        X_test = np.concatenate((X_test, img[None]), axis=0)
        y_test_subtype.append(subtype)
        y_test.append(label)

# val dataset construction
X_val = np.zeros((0, img_dims, img_dims))
y_val = []
y_val_subtype = []
for cond in ['/NORMAL/', '/PNEUMONIA/']:
    c = 0
    for img_name in (os.listdir(input_dir + 'val' + cond)):
        c = c + 1
        img = cv2.imread(input_dir + 'val' + cond + img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(img_dims, img_dims), interpolation=cv2.INTER_AREA)
        img = np.asarray(img)
        img = img / 255.

        if cond == '/NORMAL/':
            label = 0
            subtype = -1
        elif cond == '/PNEUMONIA/':
            if "bacteria" in img_name:
                label = 1
                subtype = 0
            elif "virus" in img_name:
                label = 1
                subtype = 1
            else:
                print(img_name)
        X_val = np.concatenate((X_val, img[None]), axis=0)
        y_val_subtype.append(subtype)
        y_val.append(label)

# train dataset construction
X_train = np.zeros((0, img_dims, img_dims))
y_train = []
y_train_subtype = []
for cond in ['/NORMAL/', '/PNEUMONIA/']:
    c = 0
    cv = 0
    cb = 0
    subtype = -1
    for img_name in (os.listdir(input_dir + 'train' + cond)):
        c = c + 1
        print(c)
        img = cv2.imread(input_dir + 'train' + cond + img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(img_dims, img_dims), interpolation=cv2.INTER_AREA)
        img = np.asarray(img)
        img = img / 255.

        if cond == '/NORMAL/':
            label = 0
            subtype = -1
        elif cond == '/PNEUMONIA/':
            label = 1
            if "bacteria" in img_name:
                subtype = 0
                cb += 1
            elif "virus" in img_name:
                subtype = 1
                cv += 1
            else:
                print(img_name)
        if subtype == 0 and cb < 1342:
            X_train = np.concatenate((X_train, img[None]), axis=0)
            y_train.append(1)
            y_train_subtype.append(0)
        if subtype == 1 and cv < 1342:
            X_train = np.concatenate((X_train, img[None]), axis=0)
            y_train.append(1)
            y_train_subtype.append(1)
        if subtype == -1:
            X_train = np.concatenate((X_train, img[None]), axis=0)
            y_train.append(0)
            y_train_subtype.append(-1)

"""print(X_test.shape)
print(X_val.shape)
print(X_train.shape)"""

y_train_numpy = np.array(y_train_subtype)
y_val_numpy = np.array(y_val_subtype)
y_test_numpy = np.array(y_test_subtype)

print('-------------')

print(np.sum(y_train_numpy==-1))
print(np.sum(y_train_numpy==0))
print(np.sum(y_train_numpy==1))

print('-------------')

print(np.sum(y_val_numpy==-1))
print(np.sum(y_val_numpy==0))
print(np.sum(y_val_numpy==1))

print('-------------')

print(np.sum(y_test_numpy==-1))
print(np.sum(y_test_numpy==0))
print(np.sum(y_test_numpy==1))

np.save("/home/robin/Documents/datasets/pneumonia/X_test.npy", X_test)
y_test_dict = {'diagnosis': (np.array(y_test)).tolist(), 'subtype': (np.array(y_test_subtype)).tolist()}
pd.DataFrame(y_test_dict).to_csv('/home/robin/Documents/datasets/pneumonia/y_test.csv')

np.save("/home/robin/Documents/datasets/pneumonia/X_val.npy", X_val)
y_val_dict = {'diagnosis': (np.array(y_val)).tolist(), 'subtype': (np.array(y_val_subtype)).tolist()}
pd.DataFrame(y_val_dict).to_csv('/home/robin/Documents/datasets/pneumonia/y_val.csv')

np.save("/home/robin/Documents/datasets/pneumonia/X_train.npy", X_train)
y_train_dict = {'diagnosis': (np.array(y_train)).tolist(), 'subtype': (np.array(y_train_subtype)).tolist()}
pd.DataFrame(y_train_dict).to_csv('/home/robin/Documents/datasets/pneumonia/y_train.csv')
