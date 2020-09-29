from pypads_padre.concepts.util import get_by_tag
from test.base_test import BaseTest, TEST_FOLDER


class PyPadsDecoratorsTest(BaseTest):

    def test_dataset(self):
        """
        This example will track the concepts created by the decorated function
        """
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, autostart=True)

        from sklearn.datasets import make_classification
        ds_name = "generated"

        @tracker.decorators.dataset(name=ds_name, target=[-1])
        def load_wine():
            import numpy as np
            X, y = make_classification(n_samples=150)
            data = np.concatenate([X,y.reshape(len(y),1)], axis=1)
            return data

        data = load_wine()

        # --------------------------- asserts ---------------------------
        import mlflow
        datasets_repo = mlflow.get_experiment_by_name("datasets")
        datasets = get_by_tag("pypads.dataset", experiment_id=datasets_repo.experiment_id)

        def get_name(run):
            tags = run.data.tags
            return tags.get("pypads.dataset", None)

        ds_names = [get_name(ds) for ds in datasets]
        assert ds_name in ds_names

        # !-------------------------- asserts ---------------------------

    def test_custom_splitter(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, autostart=True)

        @tracker.decorators.dataset(name="iris")
        def load_iris():
            from sklearn.datasets import load_iris
            return load_iris()

        @tracker.decorators.splitter()
        def splitter(data, training=0.6):
            import numpy as np
            idx = np.arange(data.shape[0])
            cut = int(len(idx) * training)
            return idx[:cut], idx[cut:]

        data = load_iris()

        train_idx, test_idx = splitter(data.data, training=0.7)

        # --------------------------- asserts ---------------------------
        import numpy
        assert tracker.cache.run_exists("current_split")
        split = tracker.cache.run_get("current_split")
        assert tracker.cache.run_get(split).get("split_info", None) is not None
        split_info = tracker.cache.run_get(split).get("split_info")
        train, test = split_info.get("train"), split_info.get("test")

        assert numpy.array_equal(train_idx, train)
        assert numpy.array_equal(test_idx, test)
        # !-------------------------- asserts ---------------------------

    def test_default_splitter_with_no_params(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, autostart=True)

        @tracker.decorators.dataset(name="iris")
        def load_iris():
            from sklearn.datasets import load_iris
            return load_iris()

        data = load_iris()

        splits = tracker.actuators.default_splitter(data.data)

        # --------------------------- asserts ---------------------------
        import numpy
        num = -1
        for train_idx, test_idx, val_idx in splits:
            num += 1
            print("train: {}\n test: {}\n val: {}".format(train_idx, test_idx, val_idx))
            assert tracker.cache.run_exists("current_split")
            split = tracker.cache.run_get("current_split")
            assert num == split
            assert tracker.cache.run_get(split).get("split_info", None) is not None
            split_info = tracker.cache.run_get(split).get("split_info")
            train, test, val = split_info.get("train"), split_info.get("test"), split_info.get("val")

            assert numpy.array_equal(train_idx, train)
            assert numpy.array_equal(test_idx, test)
            assert val_idx is None and val is None
        # !-------------------------- asserts ---------------------------

    def test_default_splitter_with_params(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, autostart=True)

        @tracker.decorators.dataset(name="iris")
        def load_iris():
            from sklearn.datasets import load_iris
            return load_iris()

        data = load_iris()

        splits = tracker.actuators.default_splitter(data.data, strategy="cv", n_folds=3, val_ratio=0.2)

        # --------------------------- asserts ---------------------------
        import numpy
        num = -1
        for train_idx, test_idx, val_idx in splits:
            num += 1
            print("train: {}\n test: {}\n val: {}".format(train_idx, test_idx, val_idx))
            assert tracker.cache.run_exists("current_split")
            split = tracker.cache.run_get("current_split")
            assert num == split
            assert tracker.cache.run_get(split).get("split_info", None) is not None
            split_info = tracker.cache.run_get(split).get("split_info")
            train, test, val = split_info.get("train"), split_info.get("test"), split_info.get("val")

            assert numpy.array_equal(train_idx, train)
            assert numpy.array_equal(test_idx, test)
            assert numpy.array_equal(val_idx, val)
        # !-------------------------- asserts ---------------------------

    def test_hyperparameters(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, autostart=True)

        @tracker.decorators.hyperparameters()
        def parameters():
            param1: int = 0
            param2 = "test"
            return

        parameters()

        # --------------------------- asserts ---------------------------
        assert tracker.cache.run_exists(parameters.__qualname__)
        params = tracker.cache.run_get(parameters.__qualname__)
        assert "param1" in params.keys() and "param2" in params.keys()
        assert params.get("param1") == 0 and params.get("param2") == "test"
        # !-------------------------- asserts ---------------------------

    def test_track(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, autostart=True)

        @tracker.decorators.track(event="pypads_metric")
        def roc_auc(y_test,scores, n_classes):
            from sklearn.metrics import roc_curve, auc
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), scores.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            return roc_auc

        # !-------------------------- asserts ---------------------------
