# Movie-genre-detection-from-posters
# Movie Genere prediction from posters

This project aims at determinig the genre of the movie using its posters for image classifications CNNs are the most effective types of neural network in this project we try to create a CNN which would predict the genres of these movies. 
## Dataset
We used a dataset containing 15000 movie posters which were classified into drama, thriller, action, comedy, biography or documentary or a combination of these posters were downloaded from IMDB using web scrapping and a csv file was generated containg the title and its genres.
The dataset was divided into 3 sets with 80% for training 10% for test and validation set.

![arizona](https://drive.google.com/uc?export=view&id=1Xek4IJP6EzzwrKI3HzBAdC5_KBm8DuC8)

![space](https://drive.google.com/uc?export=view&id=189gyCdljlsYHYufao8AOK0sFk9jCYfL9)

![marvel](https://drive.google.com/uc?export=view&id=1AYFCgrCmnYtuzrN271ghlpssQ2DEUew5)


## Preprocessing

The images were resized using the skimage module

```bash
img = skimage.transform.resize(img,size)
```

The opencv module can also be used to achieve the same goal of resizing
```bash
img = cv2.resize(img,size)
```

## CNN model
we used the keras module to make our convolutional neural network.the model has the architecture is given below:
1. 2 convolutional layers with 32 filters
2. Max poooling and dropout
3. 2 convolutional layer with 64 filters
4. Dense layer with 128 neurons
5. Output neuron with 6 neurons

![bldia-250](https://drive.google.com/uc?export=view&id=1lokj5jSNCKVC_C9lweikit-1zfeX36P0)

## Results
Some movie posters were fed to our trained model and the following were the outputs. There are six output neurons each one corresponding to a particular genre and each neuron gives a real number output. The three highest value for each poster is given below:


![mi-50](https://drive.google.com/uc?export=view&id=1oZrZMc20vPS2wpD_ppG-Pc5aPQ59rdgJ)



**Prediction:**

Action 0.5179066 | Crime 0.19378987 | Horror 0.2179066

![yjhd](https://drive.google.com/uc?export=view&id=1oyiX9YG9bb2UW8kHidh8-EJ6wdU4idoB)

**Prediction:**

Romance  0.5043122 | Drama 0.34689817 | Biography 0.1913521

![inception](https://drive.google.com/uc?export=view&id=1ITLehQsWwxNc22PADGgsNfSuvgqvooN7)

**Prediction:**

Thriller 0.4531901 | Drama 0.2649131 | Action 0.2391053

## Authors

- [@Pritam Suttraway](https://github.com/PritamSS)

- [@Santanu Kumar](https://github.com/santanukumar666)
