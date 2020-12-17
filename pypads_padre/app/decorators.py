from typing import Union

from pypads.app.api import ApiPluginManager, PyPadsApi
from pypads.app.decorators import IDecorators, decorator

from pypads_padre.app.api import PadrePadsApi
from pypads_padre.util import get_class_that_defined_method, get_module_that_defined_class


class PadrePadsDecorators(IDecorators):
    def __init__(self):
        super().__init__()

    @property
    def pypads(self):
        from pypads.app.pypads import get_current_pads
        return get_current_pads()

    @property
    def api(self) -> Union[ApiPluginManager, PyPadsApi, PadrePadsApi]:
        return self.pypads.api

    # ------------------------------------------- decorators --------------------------------
    @decorator
    def dataset(self, mapping=None, name=None, target_columns=None, output_format=None, metadata=None, **kwargs):
        """
        Decorator for your custom dataset loading function for automatic logging.
        :param name: Name of your given dataset.
        :param mapping: A mapping for additional injection.
        :param target_columns: indices/names of targets or labels columns in case the returned dataset is a single object.
        :param output_format: A dict describing the outputs of your custom function in case of multiple returned objects.

        Example:
            def load_data():
                X, y = make_classification(n_samples=150)
                return X,y

            For this function we have to give the output_format as follows:
                output_format = {'X': 'features', 'y': 'targets'} such that entries of the dict have the same order
                as the outputs. Keys of the dict would dictate the names of the stored binaries:
                    - Generated_Dataset_X.pickle, Generated_Dataset_y.pickle

            @tracker.decorators.dataset(name="Generated Dataset", output_format={'X': 'features', 'y': 'targets'})
            def load_data():
                X, y = make_classification(n_samples=150)
                return X,y
        """

        def track_decorator(fn):
            ctx = get_class_that_defined_method(fn)
            return self.api.track_dataset(ctx=ctx, fn=fn, name=name, target_columns=target_columns,
                                          output_format=output_format,
                                          metadata=metadata,
                                          mapping=mapping,
                                          **kwargs)

        return track_decorator

    @decorator
    def splitter(self, mapping=None):
        def track_decorator(fn):
            ctx = get_class_that_defined_method(fn)
            return self.api.track_splits(ctx=ctx, fn=fn, mapping=mapping)

        return track_decorator

    @decorator
    def hyperparameters(self, mapping=None):
        def track_decorator(fn):
            ctx = get_class_that_defined_method(fn)
            return self.api.track_parameters(ctx=ctx, fn=fn, mapping=mapping)

        return track_decorator

    @decorator
    def parameter_search(self, mapping=None):
        def track_decorator(fn):
            ctx = get_class_that_defined_method(fn)
            return self.api.track_parameter_search(ctx=ctx, fn=fn, mapping=mapping)

        return track_decorator

    #
    @decorator
    def watch(self, track="hyper-parameters", debugging=False, mappings=None):
        def track_decorator(cls):
            ctx = get_module_that_defined_class(cls)
            fn_anchors = dict()
            if track == "all":
                fn_anchors.update({"__init__": ["pypads_params"]})
                if hasattr(ctx, "forward"):
                    fn_anchors.update({"forward": ["pypads_predict"]})
            elif track == "hyper-parameters":
                fn_anchors.update({"__init__": ["pypads_params"]})
            elif track == "model" or debugging:
                fn_anchors.update({"__init__": ["pypads_model"]})
            elif track == "output":
                if hasattr(ctx, "forward"):
                    fn_anchors.update({"forward": ["pypads_predict"]})
            if fn_anchors != {}:
                return self.api.track_model(cls, ctx=ctx, fn_anchors=fn_anchors, mappings=mappings)
            else:
                return cls

        return track_decorator
