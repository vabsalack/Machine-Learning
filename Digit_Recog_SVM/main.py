import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def predict_digit(model, n, data_set):
    result = model.predict(data_set.images[n].reshape((1, -1)))
    plt.imshow(data_set.images[n],
               cmap=plt.cm.gray_r,
               interpolation="nearest")

    print(result)
    plt.axis("off")
    plt.title(f"{result}")
    plt.show()


def main():
    data_set = load_digits()
    print(dir(data_set))
    # print(data_set.DESCR)
    print(data_set.target)
    # print(data_set.data)
    # print(data_set.images)

    no_of_images = len(data_set.images)
    # print(no_of_images)

    # plt.gray()
    # plt.matshow(data_set.images[5])
    # plt.show()

    x = data_set.images.reshape((no_of_images, -1))
    y = data_set.target

    xtr, xte, ytr, yte = train_test_split(x,
                                          y,
                                          test_size=0.25,
                                          random_state=52)

    print(xtr.shape, "\n", xte.shape)

    model = svm.SVC(kernel="linear")
    model.fit(xtr, ytr)

    predict_digit(model, 13, data_set)


if __name__ == "__main__":
    main()