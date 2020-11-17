from typing import Dict

from pypads.app.api import IApi, cmd
from pypads.importext.mappings import Mapping


class PadrePadsApi(IApi):
    def __init__(self):
        super().__init__()

    @property
    def pypads(self):
        from pypads.app.pypads import get_current_pads
        return get_current_pads()

    @cmd
    def track_dataset(self, fn, ctx=None, name=None, target_columns=None, metadata=None, mapping: Mapping = None,
                      **kwargs):
        """
        Manually wrap a function to track as dataset
        """
        if metadata is None:
            metadata = {}
        self.pypads.cache.run_add('dataset_name', name)
        self.pypads.cache.run_add('dataset_metadata', metadata)
        self.pypads.cache.run_add('dataset_kwargs', {**{"target_columns": target_columns}, **kwargs})
        return self.pypads.api.track(fn, ctx, ["pypads_dataset"], mapping=mapping)

    @cmd
    def track_splits(self, fn, ctx=None, mapping: Mapping = None):
        """
        Manually wrap a function to track as split
        """
        return self.pypads.api.track(fn, ctx, ["pypads_split"], mapping=mapping)

    @cmd
    def track_parameters(self, fn, ctx=None, mapping: Mapping = None):
        """
        Manually wrap a function to track as parameter provider
        """
        return self.pypads.api.track(fn, ctx, ["pypads_params"], mapping=mapping)

    @cmd
    def track_parameter_search(self, fn, ctx=None, mapping: Mapping = None):
        """
        Manually wrap a function to track as parameter search
        """
        return self.pypads.api.track(fn, ctx, ["pypads_param_search"], mapping=mapping)

    @cmd
    def track_model(self, cls, ctx=None, fn_anchors: dict = None, mappings: Dict[str, Mapping] = None):
        """
        Manually wrap a pytorch model to track its layers, dimensions
        """
        return self.pypads.api.track_class(cls, ctx, fn_anchors, mappings=mappings)
