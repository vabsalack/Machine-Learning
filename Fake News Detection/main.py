from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import itertools

df = pd.read_csv("fake_or_real_news.csv")
df.set_index("Unnamed: 0",
             inplace=True)

y = df.label

df.drop("label",
        axis=1,
        inplace=True)

xtrain, xtest, ytrain, ytest = train_test_split(df.text,
                                                y,
                                                test_size=0.33,
                                                random_state=53)

cvectorizer = CountVectorizer(stop_words="english")
ctrain = cvectorizer.fit_transform(xtrain)
ctest = cvectorizer.transform(xtest)

temp = cvectorizer.get_feature_names()[:10]
print(temp)

count_df = pd.DataFrame(ctrain.A,
                        columns=temp)


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")


model = MultinomialNB()
model.fit(ctrain, ytrain)

prediction = model.predict(ctest)
score = metrics.accuracy_score(ytest, prediction)
print(f"accuracy: {score:0.3f}")

cm = metrics.confusion_matrix(ytest,
                              prediction,
                              labels=["FAKE", "REAL"])
plot_confusion_matrix(cm,
                      classes=["FAKE", "REAL"])
