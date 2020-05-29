# Abstraction and reasoning submission
## What is the abstraction and reasoning challenge
The abstraction and reasoning challenge is a competition to create an AI system that can solve reasoning tasks it has not seen before.
## My first approach
My first approach to solving this problem was basen on [one shot learning using a siamese neural network](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d).
What my one shot detector basically does( And this is an undersimplification ) is encode it's input task into a high dimensional vector, Using a neural network in this case. Then, by running this same network on two different task examples and comparing the distance between the resulting output vectors it can identify if they are of the same type.
