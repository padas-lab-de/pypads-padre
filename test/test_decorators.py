from pypads_padre.concepts.util import persistent_hash
from test.base_test import BaseTest, TEST_FOLDER


class PyPadsDecoratorsTest(BaseTest):

    def test_dataset(self):
        """
        This example will track the concepts created by the decorated function
        """
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(autostart=True)

        from sklearn.datasets import make_classification
        ds_name = "generated"

        @tracker.decorators.dataset(name=ds_name, output_format={'X': 'features', 'y': 'targets'})
        def load_wine():
            X, y = make_classification(n_samples=150)
            return X, y

        dataset = load_wine()

        # --------------------------- asserts ---------------------------
        datasets_repo = tracker.dataset_repository
        hash_id = persistent_hash(str(dataset))

        self.assertTrue(datasets_repo.has_object(uid=hash_id))

        # !-------------------------- asserts ---------------------------
        tracker.api.end_run()

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
        datasets_repo = tracker.dataset_repository
        hash_id = persistent_hash(str(data))
        self.assertTrue(datasets_repo.has_object(uid=hash_id))

        self.assertTrue(tracker.cache.run_exists("current_split"))
        split_id = tracker.cache.run_get("current_split")
        from pypads_padre.bindings.events import DEFAULT_PADRE_LOGGING_FNS
        SplitILF = DEFAULT_PADRE_LOGGING_FNS["splits"][0]
        _id = id(SplitILF)
        self.assertTrue(tracker.cache.run_exists(_id))
        logger_cached = tracker.cache.run_get(_id)
        output = logger_cached.get('output')
        splits = output.splits
        self.assertTrue(str(split_id) in splits.splits.keys())
        # !-------------------------- asserts ---------------------------
        tracker.api.end_run()

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
        # id of the splits logger
        from pypads_padre.bindings.events import DEFAULT_PADRE_LOGGING_FNS
        SplitILF = DEFAULT_PADRE_LOGGING_FNS["splits"][0]
        _id = id(SplitILF)

        import numpy

        for train_idx, test_idx, val_idx in splits:

            print("train: {}\n test: {}\n val: {}".format(train_idx, test_idx, val_idx))
            self.assertTrue(tracker.cache.run_exists("current_split"))
            split_id = tracker.cache.run_get("current_split")
            self.assertTrue(tracker.cache.run_exists(_id))
            logger_cached = tracker.cache.run_get(_id)
            output = logger_cached.get('output')
            splits = output.splits
            self.assertTrue(str(split_id) in splits.splits.keys())

            current_split = splits.splits[str(split_id)]

            self.assertTrue(numpy.array_equal(train_idx, current_split.train_set))
            self.assertTrue(numpy.array_equal(test_idx, current_split.test_set))
            self.assertTrue(val_idx is None and current_split.validation_set is None)
        # !-------------------------- asserts ---------------------------
        tracker.api.end_run()

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
        # id of the splits logger
        from pypads_padre.bindings.events import DEFAULT_PADRE_LOGGING_FNS
        SplitILF = DEFAULT_PADRE_LOGGING_FNS["splits"][0]
        _id = id(SplitILF)
        import numpy

        for train_idx, test_idx, val_idx in splits:
            print("train: {}\n test: {}\n val: {}".format(train_idx, test_idx, val_idx))
            self.assertTrue(tracker.cache.run_exists("current_split"))
            split_id = tracker.cache.run_get("current_split")
            self.assertTrue(tracker.cache.run_exists(_id))
            logger_cached = tracker.cache.run_get(_id)
            output = logger_cached.get('output')
            splits = output.splits
            self.assertTrue(str(split_id) in splits.splits.keys())

            current_split = splits.splits[str(split_id)]

            self.assertTrue(numpy.array_equal(train_idx, current_split.train_set))
            self.assertTrue(numpy.array_equal(test_idx, current_split.test_set))
            self.assertTrue(numpy.array_equal(val_idx, current_split.validation_set))
        # !-------------------------- asserts ---------------------------
        tracker.api.end_run()

    # def test_track(self):
    #     # --------------------------- setup of the tracking ---------------------------
    #     # Activate tracking of pypads
    #     from pypads.app.base import PyPads
    #     tracker = PyPads(uri=TEST_FOLDER, autostart=True)
    #
    #     @tracker.decorators.track(event="pypads_metric")
    #     def roc_auc(y_test, scores, n_classes):
    #         from sklearn.metrics import roc_curve, auc
    #         # Compute ROC curve and ROC area for each class
    #         fpr = dict()
    #         tpr = dict()
    #         roc_auc = dict()
    #         for i in range(n_classes):
    #             fpr[i], tpr[i], _ = roc_curve(y_test[:, i], scores[:, i])
    #             roc_auc[i] = auc(fpr[i], tpr[i])
    #
    #         # Compute micro-average ROC curve and ROC area
    #         fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), scores.ravel())
    #         roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #         return roc_auc
    #
    #     # !-------------------------- asserts ---------------------------
    #     tracker.api.end_run()
