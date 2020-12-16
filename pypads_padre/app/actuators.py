from pypads.app.actuators import IActuators, actuator

from pypads_padre.concepts.splitter import default_split
from pypads_padre.util import get_class_that_defined_method


class PadrePadsActuators(IActuators):
    def __init__(self):
        super().__init__()

    @property
    def pypads(self):
        from pypads.app.pypads import get_current_pads
        return get_current_pads()

    # noinspection PyMethodMayBeStatic
    @actuator
    def default_splitter(self, X, y=None, **kwargs):
        """
        Function to provide default splittings of pypads.
        Currently the following splitting strategies are supported:
         - random split (stratified / non-stratified). If no_shuffle is true, the order will not be changed.
         - cross validation (stratified / non-stratified)"
         - explicit - expects an explicit split given as parameter indices = (train_idx, val_idx, test_idx)
         - function - expects a function taking as input the dataset and performing the split.
         - none - there will be no splitting. only a training set will be provided

         Options:
         ========
         - strategy={"random"|"cv"|"explicit"|"none"/None} splitting strategy, default random
         - test_ratio=float[0:1]   ratio of test dataset, default 0.25
         - val_ratio=float[0:1]    ratio of the validation test (taken from the training set), default 0
         - n_folds=int             number of folds when selecting cv strategies, default 3. smaller than dataset size
         - random_seed=int         seed for the random generator or None if no seeding should be done
         - stratified={True|False|None} True, if splits should consider class stratification. If None, than stratification
                                   is activated when there are targets (default). Otherwise, stratification strategies
                                    is taking explicitly into account
         - shuffle={True|False} indicates, whether shuffling the data is allowed.
         - indices = [(train, validation, test)] a list of tuples with three index arrays in the dataset.
                                   Every index array contains
                                   the row index of the datapoints contained in the split
        :return:
        """
        ctx = get_class_that_defined_method(default_split)
        keys = list(kwargs.keys())
        for key in keys:
            if key.startswith("_pypads") or key.startswith("_logger"):
                del kwargs[key]
        return self.pypads.api.track_splits(ctx=ctx, fn=default_split, mapping=None)(X, y=y, **kwargs)

