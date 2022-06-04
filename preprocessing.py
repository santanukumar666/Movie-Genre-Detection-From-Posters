import sys
import os
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
import glob
import imageio
from sklearn.model_selection import train_test_split
import cv2
import skimage.transform
import json
import re


def process_labels(label_dict, genres):
    idx = 0
    for g_row in genres:
        for g in g_row:
            if g in label_dict["id2genre"]:
                pass
            else:
                label_dict["id2genre"].append(g)
                label_dict["genre2id"][g] = idx
                idx += 1

    n_classes = len(label_dict["id2genre"])
    print("identified {} classes".format(n_classes))
    return label_dict


def download_posters(savelocation):
    poster_not_found = []
    cnt = 1
    for index, row in df.iterrows():
        url = row['Poster']
        if "https://images-na.ssl-images-amazon.com/images/" in str(url):
            idx = row['imdbId']
            imagename = savelocation+str(idx)+".jpg"
            try:
                urlretrieve(url, imagename)
            except:
                poster_not_found.append(index)
        else:
            poster_not_found.append(index)
        print(cnt)
        cnt += 1

    df.drop(df.index[poster_not_found], inplace=True)


def get_id(imgname):
    s_index = imgname.rfind("/")+1
    e_index = imgname.rfind(".jpg")
    return imgname[s_index:e_index]


def concatenator(sl):
    str = ""
    for i in range(len(sl)):
        str = str+sl[i]

    return str


def prepare(data, posters_dict, label_dict, size):
    x = []
    y = []
    ids = []

    n_classes = len(label_dict["id2genre"])
    for idx in posters_dict.keys():
        try:
            img = skimage.transform.resize(posters_dict[idx], size)
            img = img.astype(np.float32)
            z = re.findall("\d", idx)
            num = concatenator(z)
            #data = data[data['imdbId'].notna()]

            genre = data[data['imdbId'] == int(
                num)]['Genre'].values[0].split('|')
            l = np.sum([np.eye(n_classes, dtype="uint8")[
                label_dict["genre2id"][s]] for s in genre], axis=0)
            if img.shape != size or len(l) == 0:
                continue
            #print(np.eye(n_classes, dtype="uint8"))
            x.append(img)
            y.append(l)
            ids.append(idx)
        except:
            pass
    return x, y, ids


df = pd.read_csv(
    "MovieGenre.csv", encoding="ISO-8859-1")
# print(df.head());

# assign unique id to all genres
label_dict = {"genre2id": {}, "id2genre": []}

genres = df['Genre'].apply(lambda row: str(row).split("|"))

# print(genres)
label_dict = process_labels(label_dict, genres)
# print(label_dict)
with open('label.json', 'w') as lf:
    json.dump(label_dict, lf)
n_classes = len(label_dict["id2genre"])
#savelocation = "/content/drive/MyDrive/movie genre test/"

#comment the below line after sucessful download of posters
download_posters(savelocation)


#image_glob = glob.glob(savelocation +"*.jpg")
DEST = 'movie genre test'

files = os.listdir(DEST)
# print(len(files))
posters_dict = {}
cnt = 0
for image in files:
    image2 = DEST + '/' + str(image)
    try:
        posters_dict[get_id(image2)] = imageio.imread(
            image2)  # Imageio.imread is similar to cv2.imread
        if cnt == 15000:
            break
        print(cnt)
        cnt += 1
    except:
        pass

n_samples = len(posters_dict)
print(len(posters_dict))
print("identified {} samples".format(n_samples))
# print(posters_dict.keys())
# (268,182,3)
img_size = (150, 100, 3)
x, y, ids = prepare(df, posters_dict, label_dict, size=img_size)

X = np.array(x, 'float32')
Y = np.array(y, 'float32')

print(X.shape)
print(Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
