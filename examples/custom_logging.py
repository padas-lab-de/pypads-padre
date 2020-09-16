from pypads.app.base import PyPads

tracker = PyPads(autostart=True)
# just changing stuff
from sklearn import datasets
from sklearn.metrics.classification import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

# load the iris datasets
dataset = datasets.load_iris()

# Splitting the data
splitter = KFold(n_splits=3)

# fit a model to the data
i = 0
scores = []
for train_idx, test_idx in splitter.split(dataset.data, y=dataset.target):
    print("Split number : %d" % i)
    model = DecisionTreeClassifier()
    model.fit(dataset.data[train_idx], dataset.target[train_idx])
    # make predictions
    expected = dataset.target[test_idx]
    predicted = model.predict(dataset.data[test_idx])
    # summarize the fit of the model
    print("Score: " + str(f1_score(expected, predicted, average="macro")))
    scores.append(f1_score(expected, predicted, average="macro"))

print("Average score over {} splits : {}".format(i + 1, sum(scores) / len(scores)))
tracker.api.end_run()
