import os

import cv2
import numpy as np
import pandas as pd
import torchvision
from matplotlib import pyplot as plt

input_dir = "/home/robin/Documents/datasets/retinal_OCT/"

img_dims = 224

# To create the "_disease" version of the datasets, please just comment the lines that
# concatenate the healthy labels and/or images to X_*, y_*, and y_*_subtype

# test dataset construction
X_test = np.zeros((0, img_dims, img_dims))
y_test = []
y_test_subtype=[]
test_list_of_patient_ids = []
for diagnosis, cond in enumerate(['NORMAL/', 'DME/', 'DRUSEN/', 'CNV/', ]):  #
    print(cond)
    for img_name in (os.listdir(input_dir + 'test/' + cond)):
        patient_id = img_name.split("-")[1]
        test_list_of_patient_ids.append(patient_id)
        img = cv2.imread(input_dir + 'test/' + cond + img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(img_dims, img_dims), interpolation=cv2.INTER_AREA)
        img = np.asarray(img)
        img = img / 255.
        X_test = np.concatenate((X_test, np.array(img)[None]), axis=0)
        if diagnosis == 0 :
            y_test.append(diagnosis)
        else :
            y_test.append(1)
        y_test_subtype.append(diagnosis-1)

# val dataset construction
X_val = np.zeros((0, img_dims, img_dims))
y_val = []
y_val_subtype = []
for diagnosis, cond in enumerate(['NORMAL/', 'DME/', 'DRUSEN/', 'CNV/', ]):  #
    print(cond)
    for img_name in (os.listdir(input_dir + 'val/' + cond)):
        patient_id = img_name.split("-")[1]
        test_list_of_patient_ids.append(patient_id)
        img = cv2.imread(input_dir + 'val/' + cond + img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(img_dims, img_dims), interpolation=cv2.INTER_AREA)
        img = np.asarray(img)
        img = img / 255.
        X_val = np.concatenate((X_val, np.array(img)[None]), axis=0)
        if diagnosis == 0 :
            y_val.append(diagnosis)
        else :
            y_val.append(1)
        y_val_subtype.append(diagnosis-1)

# train dataset construction
X_train = np.zeros((0, img_dims, img_dims))
y_train = []
y_train_subtype = []
list_of_patient_ids = []
count_of_patient_ids = []
for diagnosis, cond in enumerate(['NORMAL/', 'DME/', 'DRUSEN/', 'CNV/']) :
    c = 0
    print(cond)
    for img_name in (os.listdir(input_dir + 'train/' + cond)):
        patient_id = img_name.split("-")[1]
        if patient_id in test_list_of_patient_ids:
            continue
        if patient_id in list_of_patient_ids:
            if count_of_patient_ids[list_of_patient_ids.index(patient_id)] == 4 :
                continue
            else :
                count_of_patient_ids[list_of_patient_ids.index(patient_id)] += 1
        else:
            list_of_patient_ids.append(patient_id)
            count_of_patient_ids.append(1)
        img = cv2.imread(input_dir + 'train/' + cond + img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(img_dims, img_dims), interpolation=cv2.INTER_AREA)
        img = np.asarray(img)
        img = img / 255.
        if cond == 'NORMAL/' and c == (3 * 1000):
            continue
        if cond != 'NORMAL/' and c == 1000:
            continue
        c += 1
        X_train = np.concatenate((X_train, np.array(img)[None]), axis=0)
        if diagnosis == 0 :
            y_train.append(diagnosis)
        else:
            y_train.append(1)
        y_train_subtype.append(diagnosis-1)

print(X_test.shape)
print(X_val.shape)
print(X_train.shape)

y_val_numpy = np.array(y_val)
y_train_numpy = np.array(y_train)
y_test_numpy = np.array(y_test)

print('-------------')

print(np.sum(y_val_numpy==0))
print(np.sum(y_val_numpy==1))
print(np.sum(y_val_numpy==2))
print(np.sum(y_val_numpy==3))

print('-------------')

print(np.sum(y_train_numpy==0))
print(np.sum(y_train_numpy==1))
print(np.sum(y_train_numpy==2))
print(np.sum(y_train_numpy==3))

print('-------------')

print(np.sum(y_test_numpy==0))
print(np.sum(y_test_numpy==1))
print(np.sum(y_test_numpy==2))
print(np.sum(y_test_numpy==3))


np.save("/home/robin/Documents/datasets/retinal_OCT/X_test.npy", X_test)
y_test_dict = {'diagnosis': y_test, 'subtype': y_test_subtype}
pd.DataFrame(y_test_dict).to_csv('/home/robin/Documents/datasets/retinal_OCT/y_test.csv')

np.save("/home/robin/Documents/datasets/retinal_OCT/X_val.npy", X_val)
y_val_dict = {'diagnosis': y_val, 'subtype': y_val_subtype}
pd.DataFrame(y_val_dict).to_csv('/home/robin/Documents/datasets/retinal_OCT/y_val.csv')

np.save("/home/robin/Documents/datasets/retinal_OCT/X_train.npy", X_train)
y_train_dict = {'diagnosis': y_train, 'subtype': y_train_subtype}
pd.DataFrame(y_train_dict).to_csv('/home/robin/Documents/datasets/retinal_OCT/y_train.csv')
