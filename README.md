# Abstraction and reasoning submission
## What is the abstraction and reasoning challenge
The abstraction and reasoning challenge is a competition to create an AI system that can solve reasoning tasks it has not seen before.
## My first approach
My first approach to solving this problem was basen on [one shot learning using a siamese neural network](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d).

What my one shot detector basically does is get given an example of the problem and solution of a task and encode this into a high dimensional vector, Using a neural network in this case. Then, by running this same network on two different problem-solution pairs and comparing the distance between the resulting output vectors it can identify if they are of the same type.

After training this detector to be able to distinguish between different tasks, I then ran something similar to an [adversarial attack](https://openai.com/blog/adversarial-example-research/) on it. First, I ran it on all available.
