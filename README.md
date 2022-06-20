# Linear-Regression
NYCU, Pattern Recognition, homework1 

This project is to implement linear regression by using only NumPy with Gradient Descent.

The sample code can be download in this [link](https://github.com/NCTU-VRDL/CS_AT0828/tree/main/HW1).

## Requirement

In this work, you can use the following commend to build the environment.

```bash
$ conda create --name PR python=3.8 -y
$ conda activate PR
$ conda install matplotlib pandas -y
$ pip install tqdm
```

## Training & Evaluation 

You can use the following commend and select the option to train specify model. After training the model, the program will automatic evaluate the model.

```bash
python 310551031_hw1.py
```

## Result

The evaluation metrics is Mean Sequre Error.

|     | Gradient descent | Minibatch Gradient Descent | Stochastic Gradient Descent |
|-----|------------------|----------------------------|-----------------------------|
| MSE | 0.0083           | 0.04124                    | 0.0399                      |


