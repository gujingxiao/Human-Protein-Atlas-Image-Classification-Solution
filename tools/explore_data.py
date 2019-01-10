import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imread

import tensorflow as tf
sns.set()

import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Any results you write to the current directory are saved as output.
# train_labels1 = pd.read_csv("../HumanProteinData/train.csv")
# train_labels2 = pd.read_csv("./external_data_select.csv")

# train_labels1.to_csv("./combine_data.csv", index=False)
# train_labels2.to_csv("./combine_data.csv", index=False, header=False, mode='a+')
train_labels = pd.read_csv("../HumanProteinData/combine_data.csv")
train_labels.head()
print(train_labels.shape[0])

label_names = {
    0:  "Nucleoplasm",  # 0
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",
    12:  "Actin filaments",
    13:  "Focal adhesion sites",
    14:  "Microtubules",
    15:  "Microtubule ends",
    16:  "Cytokinetic bridge",
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",
    19:  "Centrosome",
    20:  "Lipid droplets",
    21:  "Plasma membrane",
    22:  "Cell junctions",
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",  # 25
    26:  "Cytoplasmic bodies",
    27:  "Rods & rings"
}

reverse_train_labels = dict((v,k) for k,v in label_names.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row

for key in label_names.keys():
    train_labels[label_names[key]] = 0

train_labels = train_labels.apply(fill_targets, axis=1)
train_labels.head()

target_counts = train_labels.drop(["Id", "Target"],axis=1).sum(axis=0).sort_values(ascending=False)
print(target_counts)
# plt.figure(figsize=(15,15))
# sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)
# plt.show()