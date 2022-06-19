import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# scale feature into [0,1] through Mean Normalization (accelerate diverge)
def normalize(x):
    return (x - np.mean(x, axis=0))/(np.max(x, axis=0) - np.min(x, axis=0))


def criterion(y_hat, y, m):  # calculate the loss through Mean Square Error
    return 1/(2*m) * np.sum((y_hat - y)**2)


def compute_gradient(x, y_hat, y, m):  # Gradient Descent
    return 1/m * np.sum((y_hat-y)*x, axis=0).reshape(-1, 1)


def create_dir(path):  # create directory that can save image
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return None


def data_prepare(csv_name, norm):  # pack input into several batch (dataloader)

    x_data, y_data = [], []

    # loading training data into several batch
    if "train" in csv_name:
        df_train = pd.read_csv(csv_name)
        x, y = df_train['x_train'].to_numpy(), df_train['y_train'].to_numpy()

    # loading testing data into several batch
    elif "test" in csv_name:
        df_test = pd.read_csv(csv_name)
        x, y = df_test['x_test'].to_numpy(), df_test['y_test'].to_numpy()

    plot_x = x
    plot_y = y

    # feature scaling
    if norm == "y":
        x = normalize(x)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x = np.hstack((np.ones(x.shape), x))

    for curr_batch in range(0, len(x), batch_size):
        x_data.append(x[curr_batch:curr_batch+batch_size])
        y_data.append(y[curr_batch:curr_batch+batch_size])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    batch = list(zip(x_data, y_data))

    return plot_x, plot_y, batch


if __name__ == "__main__":

    mode = int(input("\nInput 1 or 2 or 3(Choose the type of Gradient " +
                     "Descent)\n\n1.Gradient descent\n2.Minibatch Gradient" +
                     "Descent\n3.Stochastic Gradient Descent\n\n"))

    # Gradient descent mode
    if mode == 1:
        lr = 1e-1
        epochs = 1000
        batch_size = 500
        name = "Gradient descent"

    # Minibatch Gradient Descent mode
    elif mode == 2:
        lr = 1e-1
        epochs = 500
        batch_size = 20
        name = "Minibatch Gradient Descent"

    # Stochastic Gradient Descent mode
    elif mode == 3:
        lr = 1e-2
        epochs = 200
        batch_size = 1
        name = "Stochastic Gradient Descent"

    else:
        print("\nYour input in unexpect.")
        print("Please execute the program and input the number on 1~3.")

    normalize_agree = input("\nDo you want to normalize the input data(y/n):")

    if normalize_agree != "y" and normalize_agree != "n":
        print("\nYour input in unexpect.")
        print("The default normalize is n.")
        normalize_agree = "n"

    if mode == 1 or mode == 2 or mode == 3:

        # storage minimum loss
        min_loss = 1000

        # random the initial weight and bias(storage best weight)
        best_theta = np.random.randn(2, 1)
        train_loss_history, test_loss_history = [], []
        train_total_pred, test_total_pred = [], []

        directory_path = name+" Image"

        # random the initial weight and bias(storage weight)
        theta = np.random.randn(2, 1)
        create_dir(directory_path)

        x_train_point, y_train_point, train_batch = data_prepare(
            "train_data.csv", normalize_agree)
        x_test_point, y_test_point, test_batch = data_prepare(
            "test_data.csv", normalize_agree)

        for epoch in range(epochs):

            train_batch_loss, test_batch_loss = [], []

            # training phase(calculate train loss)
            for x_batch, y_batch in tqdm(train_batch):

                # something like y_pred = model(x)
                y_pred = np.matmul(x_batch, theta)

                # calculate mse between prediction and ground truth
                loss = criterion(y_pred, y_batch, batch_size)

                # calculate the gradient and do backpropagation
                theta -= lr * \
                    compute_gradient(x_batch, y_pred, y_batch, batch_size)

                # save the loss value to plot the figure
                train_batch_loss.append(loss)
                train_loss_history.append(loss)

                # save the best weight and bias
                if min_loss > loss:
                    best_theta = theta

            # testing phase(calculate test loss)
            for x_batch, y_batch in tqdm(test_batch):

                # like y_pred = model(x)
                y_pred = np.matmul(x_batch, best_theta)

                # calculate mse between prediction and ground truth
                loss = criterion(y_pred, y_batch, batch_size)

                # save the loss value to plot the figure
                test_batch_loss.append(loss)
                test_loss_history.append(loss)

            print("epoch "+str(epoch)+", train loss:",
                  str(sum(train_batch_loss) /
                      len(train_batch_loss))+", test loss:",
                  sum(test_batch_loss)/len(test_batch_loss))

        # training phase(calculate train predict result) --> find fitting line
        for x_batch, y_batch in train_batch:
            y_pred = np.matmul(x_batch, best_theta)
            y_pred = list(y_pred.reshape(-1))
            train_total_pred += y_pred

        # testing phase(calculate test predict result) --> find fitting line
        for x_batch, y_batch in test_batch:
            y_pred = np.matmul(x_batch, best_theta)
            y_pred = list(y_pred.reshape(-1))
            test_total_pred += y_pred

        print("The "+name+" best weight of the linear model is",
              best_theta[1][0], ",and the intercepts is", best_theta[0][0])

        # plot training data point and fitting curve
        train_point_figure = plt.figure()
        plt.title(name+" Traininig data point and Fitting Curve")
        plt.plot(x_train_point, y_train_point, '.')
        plt.plot(x_train_point, train_total_pred)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(["data point", "line"], loc="best")
        train_point_figure.savefig(directory_path+"/training_point.jpg")

        # plot testing data point and fitting curve
        test_point_figure = plt.figure()
        plt.title(name+" Testinig data point and Fitting Curve")
        plt.plot(x_test_point, y_test_point, '.')
        plt.plot(x_test_point, test_total_pred)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(["data point", "line"], loc="best")
        test_point_figure.savefig(directory_path+"/testing_point.jpg")

        # plot loss curve
        loss_figure = plt.figure()
        plt.title(name+" loss curve")
        plt.plot(train_loss_history)
        plt.plot(test_loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(["train loss", "test loss"], loc="best")
        loss_figure.savefig(directory_path+"/loss.jpg")
