# JoYuris-Classifier

It is a program that classifies members of Joyuris using deep learning.

## 1. What is 'JoYuris'?

Joyuris means members of the IZ*ONE who called Jo Yuri.
If you're wondering what this means, look at the photo below.

<image src="https://raw.githubusercontent.com/inerplat/JoYuris-Classifier/master/0.%20Document/images/JoYuris.jpg" width="50%" height="50%"/>

In the early days of their debut, reporters often misrepresented themselves because they couldn't distinguish them.

The members of Joyuris are composed of Yuri, Chaewon and Yena. At first glance, they are similar, but each has its own characteristics. If you look closely, you can tell the difference

## 2. How does it work?

This program works by using CNN (Convolutional Neural Network), a type of deep learning algorithm.

Before using CNN, it goes through a pre-treatment process to classify members of JoYuris only by their faces, including their hair.

The program uses other completed libraries (or APIs) to cut faces, inflate data through various transformations of images, and then use CNN to learn.
