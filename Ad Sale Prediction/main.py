from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import pandas as pd


def load_ds(filename):
    return pd.read_csv(filename)


def predictByValues(modell, scalar):
    age_salary = [[int(input(">Enter the age: ")), int(input(">Enter the salary: "))]]

    if modell.predict(scalar.transform(age_salary)):
        print("> He would buy..")
    else:
        print("> He would not buy..")


def main():
    ds = load_ds("DigitalAd_dataset.csv")
    # print(ds.shape, "\n", ds.head(5))

    x = ds.iloc[:, :-1].values # .values return 2d np array of values
    y = ds.iloc[:, -1].values

    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=5)

    sc = StandardScaler()
    xtr = sc.fit_transform(xtr)
    xte = sc.transform(xte)
    # print(xtr)
    # print(xte)

    model = LogisticRegression(random_state=5)
    model.fit(xtr, ytr)

    ypred = model.predict(xte)
    # print(np.concatenate((ypred.reshape(len(ypred), 1), yte.reshape(len(yte), 1)), 1))

    cm = confusion_matrix(yte, ypred)
    print(">confusion matrix", cm, sep="\n")
    print(f"Accuracy score: {accuracy_score(yte, ypred)*100}")

    return model, sc


if __name__ == "__main__":
    model, sca = main()
    predictByValues(model, sca)


