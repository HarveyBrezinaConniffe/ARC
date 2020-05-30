# Abstraction and reasoning submission
## What is the abstraction and reasoning challenge
The abstraction and reasoning challenge is a competition to create an AI system that can solve reasoning tasks it has not seen before.
It consists of a number of different 'tasks', Each task is represented as colored dots on a grid. For example:
![Example of a task](https://github.com/HarveyBrezinaConniffe/ARC/blob/master/images/demo.png)

As a human you can pretty quickly identify that the object of this task is to take the input and hollow out the rectangles. Given a new input such as this one you could probably predict the correct output:
![New Input](https://github.com/HarveyBrezinaConniffe/ARC/blob/master/images/input.png)

With a little time you could probably write a program to solve this task. However, The object of this challenge is to create a system that, given a number of examples( Consisting of input-output pairs, Such as the first image ) can learn what this task is and can then go on to solve a new arbitrary input( The second image ). This is a much more complex task.
To learn more about the abstraction and reasoning challenge visit the repository [here](https://github.com/fchollet/ARC).

## My first approach
My first approach to solving this problem was based on [one shot learning using a siamese neural network](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d).

### How it works
There are a number of steps, The first one involves learning to identify types of tasks as follows.
1. The detector is given a task example as an input-output pair.
2. This example is run through a convolutional neural network( CNN ).
3. This network outputs a high dimensional vector.
4. This network is trained on all available examples of all tasks, It's objective is to output similar vectors for examples of the same task.
5. After some training, By comparing the output vectors produced from any two input examples we can tell if they are of the same task or not.
![Diagram 1](https://github.com/HarveyBrezinaConniffe/ARC/blob/master/images/diag1.png)

After training this detector is able to distinguish between different tasks, I then ran something similar to an [adversarial attack](https://openai.com/blog/adversarial-example-research/) on it:
1. A task is chosen to solve.
2. The detector is ran on all available input-output example pairs given.
3. The resulting vectors are averaged.
4. A new input-output pair is constructed. The input is the test input given to me, The output is randomly initialized.
5. This new pair is now run through the network and we get the difference between it's output vector and the average vector we just computed.
6. At first this difference is large, As the second pair is half random. Our goal is to solve the second task, Making it no longer random and minimizing the difference between it's vector and the averaged vector.
7. We now repeatedly run gradient descent( Using the ADAM optimizer ) *on the randomly initialized task solution* with the goal of minimizing the difference between the two vectors.
8. The goal of this is to bring the difference to zero meaning that the neural network identifies the two pairs as the same task. As the program can only adjust the solution to the task( The output part of the pair ), The hope is that it will come up with a correct solution.
