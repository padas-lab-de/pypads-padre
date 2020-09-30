from pypads.app.base import PyPads
from pypads.injections.setup.hardware import ICpuRSF
tracker = PyPads(setup_fns=[ICpuRSF()])
tracker.start_track(experiment_name="Pypads-padre")
import numpy as np

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer

# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA()
# set the tolerance to a large value to make the example faster
logistic = LogisticRegression(solver='liblinear', max_iter=10000, tol=0.1, multi_class="ovr")
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

X_digits, y_digits = datasets.load_digits(return_X_y=True)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': [5, 15, 30, 45, 64],
    'logistic__C': np.logspace(-4, 4, 4),
}
search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=4, scoring=make_scorer(f1_score, average="micro"))
search.fit(X_digits, y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
search.predict(X_digits)

tracker.api.end_run()
