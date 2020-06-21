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
    def track_dataset(self, fn, ctx=None, name=None, metadata=None, mapping: Mapping = None, **kwargs):
        if metadata is None:
            metadata = {}
        self.pypads.cache.run_add('dataset_name', name)
        self.pypads.cache.run_add('dataset_meta', metadata)
        self.pypads.cache.run_add('dataset_kwargs', kwargs)
        return self.pypads.api.track(fn, ctx, ["pypads_dataset"], mapping=mapping)

    @cmd
    def track_splits(self, fn, ctx=None, mapping: Mapping = None):
        return self.pypads.api.track(fn, ctx, ["pypads_split"], mapping=mapping)

    @cmd
    def track_parameters(self, fn, ctx=None, mapping: Mapping = None):
        return self.pypads.api.track(fn, ctx, ["pypads_params"], mapping=mapping)

    @cmd
    def track_parameter_search(self, fn, ctx=None, mapping: Mapping = None):
        return self.pypads.api.track(fn, ctx, ["pypads_param_search"], mapping=mapping)
