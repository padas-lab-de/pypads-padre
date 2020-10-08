from pypads.app.backends.repository import Repository


class DatasetRepository(Repository):

    def __init__(self, *args, **kwargs):
        """
        Repository holding all the relevant schema information
        :param args:
        :param kwargs:
        """
        super().__init__(*args, name="pypads_datasets", **kwargs)


class EstimatorRepository(Repository):

    def __init__(self, *args, **kwargs):
        """
        Repository holding all the relevant schema information
        :param args:
        :param kwargs:
        """
        super().__init__(*args, name="pypads_estimators", **kwargs)
