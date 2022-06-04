
# Movie Genere prediction from posters

This project aims at determinig the genre of the movie using its posters for image classifications CNNs are the most effective types of neural network in this project we try to create a CNN which would predict the genres of these movies. 
## Dataset
We used a dataset containing 15000 movie posters which were classified into drama, thriller, action, comedy, biography or documentary or a combination of these posters were downloaded from IMDB using web scrapping and a csv file was generated containg the title and its genres.
The dataset was divided into 3 sets with 80% for training 10% for test and validation set.

![arizona](https://user-images.githubusercontent.com/60546202/171791915-07eeed85-b1ee-40fb-8250-95f267075f44.jpeg)

![space](https://user-images.githubusercontent.com/60546202/171791910-65cd70fd-4adf-4cc1-91c0-5a66736d0a6c.jpeg)

![captain](https://user-images.githubusercontent.com/60546202/171791903-84b6a199-30e8-4ab6-8d5e-8e088f7eebd6.jpeg)


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
We used the keras module to make our convolutional neural network.the model has the architecture is given below:
1. 2 convolutional layers with 32 filters
2. Max poooling and dropout
3. 2 convolutional layer with 64 filters
4. Dense layer with 128 neurons
5. Output neuron with 6 neurons

![cnn](https://user-images.githubusercontent.com/60546202/171791899-41d72f63-c889-4cc5-a1f9-50810ece8b1f.jpeg)

## Results
Some movie posters were fed to our trained model and the following were the outputs. There are six output neurons each one corresponding to a particular genre and each neuron gives a real number output. The three highest value for each poster is given below:


![mi 5](https://user-images.githubusercontent.com/60546202/171791896-016d685e-be17-4c50-887b-dba507d639a5.jpeg)

**Prediction:**

Action 0.5179066 | Crime 0.19378987 | Horror 0.2179066

![yjhd](https://user-images.githubusercontent.com/60546202/171791890-9adfce65-ca02-4433-a703-1ffe7de284a1.jpeg)

**Prediction:**

Romance  0.5043122 | Drama 0.34689817 | Biography 0.1913521

![inception](https://user-images.githubusercontent.com/60546202/171791881-ac4f744f-9824-4037-8f12-25c765001e1c.jpeg)

**Prediction:**

Thriller 0.4531901 | Drama 0.2649131 | Action 0.2391053
## Webapp
Made a simple flask webapp for demonstration.

![1](https://user-images.githubusercontent.com/60546202/171986421-045fcc0f-9e2a-4236-886c-bdf2bf640af6.jpg)

## Authors

- [@Pritam Suttraway](https://github.com/PritamSS)

- [@Santanu Kumar](https://github.com/santanukumar666)
