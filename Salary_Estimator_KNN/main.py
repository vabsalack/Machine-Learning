import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score


def load_dataset(filename):
    ds = pd.read_csv(filename)
    return ds


def calculate_cm(model, x_test, ytest):
    ypred = model.predict(x_test)
    cm = confusion_matrix(ytest, ypred)
    print("Confusion Matrix: ")
    print(cm)

    print(f"Accuracy of the model: {accuracy_score(ytest, ypred)*100}")


def model_train(data):
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    xtr, xte, ytr, yte = train_test_split(x,
                                          y,
                                          test_size=0.25,
                                          random_state=52)
    # print(type(xtr), xtr)
    sc = StandardScaler()
    xtr = sc.fit_transform(xtr)
    xte = sc.transform(xte)

    error = []
    for i in range(1, 41):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(xtr, ytr)
        prediction_i = model.predict(xte)
        error.append((i, np.mean(prediction_i != yte)))
    # error1 = [i[1] for i in error]
    #
    # plt.plot(range(1, 41),
    #          error1,
    #          color="red",
    #          linestyle="dashed",
    #          marker="o",
    #          markerfacecolor="blue",
    #          markersize=10)
    #
    # plt.title("Error rate of k[1 - 41] value")
    # plt.xlabel("K value")
    # plt.ylabel("Mean Error")
    # plt.show()

    i = sorted(error, key=lambda z: z[1])[0][0]
    model = KNeighborsClassifier(n_neighbors=i,
                                 metric="minkowski",
                                 p=2,
                                 n_jobs=-1)

    model.fit(xtr, ytr)

    return model, sc, xte, yte


def predict_data(model, sc):
    x = [[int(input(">Employee age: ")),
         int(input(">Employee Edu no: ")),
         int(input(">Employee Capital Gain: ")),
         int(input(">Employee HPW: "))]]

    x = sc.transform(x)
    return model.predict(x)


def main():
    data = load_dataset("salary.csv")
    print(data.shape)
    print(data.head(5))

    data["income"] = data["income"].map({"<=50K": 1, ">50K": 0}).astype(int)
    print(data.head)

    model, sc, xte, yte = model_train(data)

    # if predict_data(model, sc):
    #     print("Employee might not got  Salary above 50K")
    # else:
    #     print("Employee might got  Salary above 50K")

    calculate_cm(model, xte, yte)


if __name__ == "__main__":
    main()