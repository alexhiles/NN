# Implementation of an artificial neural network
General artificial neural network (ANN) implemented in Python.

Currently implemented:
- Naive Implementation based on work in the [paper](https://epubs.siam.org/doi/pdf/10.1137/18M1165748).
- General Implementation of arbitrary neurons and arbitrary layers for a
classification problem involving noughts and crosses.

## Test Case

Here we test the code for an example case using the generalized code. Each epoch represents one loop over the training data, which is typically random. The random draws can be either be with replacement (chance of using same training data in the same epoch) or without replacement. We compare both in this example.

Run

```
python ANN.py
```
in terminal

Output:

![demonstration.png](demonstration.png)


Similar loss curves are found when running the naiveANN.py file, though this is for the "with replacement" case and is doing single step updates i.e. no pre-defined epochs.  

## To Do

- Implement different loss functions (cross entropy)

- Generalize code so that inputs and outputs can be of any length for training

- Implement batch stochastic gradient descent and other optimizer techniques i.e. ADAM

- Add visualization of classifying the data points into categories

#### References

Higham, C.F. and Higham, D.J., 2019. Deep learning: An introduction for applied mathematicians. SIAM Review, 61(4), pp.860-891.
