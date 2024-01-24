import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import shutil
import json
import random
import re
import pandas as pd
from pathlib import Path
import csv

cwd = os.getcwd()
data_path = '/home/michalislazarou/PhD/filelists/tiered_dataset'
savedir = './'
dataset_list = ['base', 'val', 'novel']
base_dest = '/home/michalislazarou/PhD/Realistic_Transductive_Few_Shot-master/data/tiered_imagenet/'

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

cl = -1
folderlist = []
data_path = Path(data_path)
subdirs = [f for f in data_path.iterdir() if f.is_dir()]

img_paths= []
cls = []
for f in subdirs:
    imgs = [x for x in f.iterdir()]
    # imgs_string = imgs.stem()
    # c = imgs_string.split('/ ')
    for img in imgs:
        img_paths.append(img)
        cls.append(img.stem)
# with open('some.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(zip(img_paths, cls))
df = pd.DataFrame()
df['paths'] = img_paths
df['classes'] = cls
df.to_csv('tiered_test.csv', index=False)
    # print(len(imgs))
