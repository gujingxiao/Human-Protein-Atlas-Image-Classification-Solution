from PIL import Image
import cv2
import six
import imagehash
import os
import pandas as pd
from numpy import *

data_path = '/home/gujingxiao/projects/HumanProteinData/'
train_path = '/home/gujingxiao/projects/HumanProteinData/train/'
test_path = '/home/gujingxiao/projects/HumanProteinData/test/'
train_csv = 'external_data_select.csv'
test_csv = 'sample_submission.csv'

train_list = pd.read_csv(os.path.join(data_path, train_csv))
test_list = pd.read_csv(os.path.join(data_path, test_csv))

print('Train Number: {}, Test Number: {}'.format(len(train_list),len(test_list)))

dhash_green = []
dhash_red = []
dhash_blue = []
avghash_green = []
avghash_red = []
avghash_blue = []
count = 0
hash_size = 8

TEST = 0
TRAIN = 0
COMPARE = 1

if TEST == 1:
    for test_item in test_list['Id']:
        count += 1
        test_green = Image.open(os.path.join(test_path, test_item + '_green.png'))
        # test_red = Image.open(os.path.join(test_path, test_item + '_red.png'))
        # test_blue = Image.open(os.path.join(test_path, test_item + '_blue.png'))

        # Difference Hashing
        dhash_green_test = imagehash.dhash(test_green, hash_size=hash_size)
        # dhash_red_test = imagehash.dhash(test_red, hash_size=hash_size)
        # dhash_blue_test = imagehash.dhash(test_blue, hash_size=hash_size)
        dhash_green.append(dhash_green_test)
        # dhash_red.append(dhash_red_test)
        # dhash_blue.append(dhash_blue_test)

        # Average Hashing
        avghash_green_test = imagehash.average_hash(test_green, hash_size=hash_size)
        # avghash_red_test = imagehash.average_hash(test_red, hash_size=hash_size)
        # avghash_blue_test = imagehash.average_hash(test_blue, hash_size=hash_size)
        avghash_green.append(avghash_green_test)
        # avghash_red.append(avghash_red_test)
        # avghash_blue.append(avghash_blue_test)

        if count % 100 == 0:
            print(count)
    test_df = pd.DataFrame({'Id': test_list['Id'], 'dhash_green': dhash_green, 'avghash_green': avghash_green})
    # test_df = pd.DataFrame({'Id': test_list['Id'], 'dhash_green': dhash_green, 'dhash_red': dhash_red, 'dhash_blue': dhash_blue,
    #                         'avghash_green': avghash_green, 'avghash_red': avghash_red, 'avghash_blue': avghash_blue})

    test_df.to_csv(os.path.join(data_path, 'test_hash.csv'), index=False)

if TRAIN == 1:
    for train_item in train_list['Id']:
        count += 1
        train_green = Image.open(os.path.join(train_path, train_item + '_green.png'))
        # train_red = Image.open(os.path.join(train_path, train_item + '_red.png'))
        # train_blue = Image.open(os.path.join(train_path, train_item + '_blue.png'))

        # Difference Hashing
        dhash_green_train = imagehash.dhash(train_green, hash_size=hash_size)
        # dhash_red_train = imagehash.dhash(train_red, hash_size=hash_size)
        # dhash_blue_train = imagehash.dhash(train_blue, hash_size=hash_size)
        dhash_green.append(dhash_green_train)
        # dhash_red.append(dhash_red_train)
        # dhash_blue.append(dhash_blue_train)

        # Average Hashing
        avghash_green_train = imagehash.average_hash(train_green, hash_size=hash_size)
        # avghash_red_train = imagehash.average_hash(train_red, hash_size=hash_size)
        # avghash_blue_train = imagehash.average_hash(train_blue, hash_size=hash_size)
        avghash_green.append(avghash_green_train)
        # avghash_red.append(avghash_red_train)
        # avghash_blue.append(avghash_blue_train)

        if count % 100 == 0:
            print(count)
    train_df = pd.DataFrame({'Id': train_list['Id'], 'dhash_green': dhash_green, 'avghash_green': avghash_green})
    # train_df = pd.DataFrame({'Id': test_list['Id'], 'dhash_green': dhash_green, 'dhash_red': dhash_red, 'dhash_blue': dhash_blue,
    #                         'avghash_green': avghash_green, 'avghash_red': avghash_red, 'avghash_blue': avghash_blue})

    train_df.to_csv(os.path.join(data_path, 'train_hash.csv'), index=False)


if COMPARE == 1:
    test_hash_csv = pd.read_csv(os.path.join(data_path, 'test_hash.csv'))
    train_hash_csv = pd.read_csv(os.path.join(data_path, 'train_hash.csv'))

    for index in range(len(test_hash_csv['Id'])):
        dhash_green_test = imagehash.hex_to_hash(test_hash_csv['dhash_green'][index])
        avghash_green_test = imagehash.hex_to_hash(test_hash_csv['avghash_green'][index])
        print(index, test_hash_csv['Id'][index])
        for idx in range(len(train_hash_csv['Id'])):
            dhash_green_train = imagehash.hex_to_hash(train_hash_csv['dhash_green'][idx])
            avghash_green_train = imagehash.hex_to_hash(train_hash_csv['avghash_green'][idx])

            dhash_sim_green = 1 - (dhash_green_test - dhash_green_train) / len(dhash_green_test.hash) ** 2
            avghash_sim_green = 1 - (avghash_green_test - avghash_green_train) / len(avghash_green_test.hash) ** 2
            avg_sim = (dhash_sim_green + avghash_sim_green) / 2.0

            if avg_sim >= 0.85:
                print(idx, 'Similarity {}'.format(avg_sim))
                test_green = cv2.imread(os.path.join(test_path, test_hash_csv['Id'][index] + '_green.png'))
                test_red = cv2.imread(os.path.join(test_path, test_hash_csv['Id'][index] + '_red.png'))
                train_green = cv2.imread(os.path.join(train_path, train_hash_csv['Id'][idx] + '_green.png'))
                train_red = cv2.imread(os.path.join(train_path, train_hash_csv['Id'][idx] + '_red.png'))
                cv2.imshow('test_green', test_green)
                cv2.imshow('train_green', train_green)
                cv2.imshow('test_red', test_red)
                cv2.imshow('train_red', train_red)
                cv2.waitKey(0)