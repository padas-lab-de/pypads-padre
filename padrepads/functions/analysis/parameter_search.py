from pypads.functions.analysis.call_tracker import LoggingEnv
from pypads.functions.loggers.base_logger import LoggingFunction
from pypads.util import is_package_available


class ParameterSearch(LoggingFunction):

    def __pre__(self, ctx, *args, _pypads_env: LoggingEnv, **kwargs):
        from pypads.pypads import get_current_pads
        from padrepads.base import PyPadrePads
        pads: PyPadrePads = get_current_pads()
        pads.cache.add("parameter_search", ctx)
        # TODO save parameter grid used for the search

    def __post__(self, ctx, *args, _pypads_env: LoggingEnv, _pypads_result, **kwargs):
        from pypads.pypads import get_current_pads
        from padrepads.base import PyPadrePads
        pads: PyPadrePads = get_current_pads()

        pads.cache.pop("parameter_search")

        if is_package_available("sklearn"):
            from sklearn.model_selection._search import BaseSearchCV
            if isinstance(ctx, BaseSearchCV):
                # TODO Write general information we can extract from base search
                from sklearn.model_selection import GridSearchCV
                if isinstance(ctx, GridSearchCV):
                    # TODO Write information we can extract from GridSearchCV
                    pass


class ParameterSearchExecutor(LoggingFunction):

    def __pre__(self, ctx, *args, **kwargs):
        pass

    def __post__(self, ctx, *args, **kwargs):
        pass

    def call_wrapped(self, ctx, *args, _pypads_env: LoggingEnv, _args, _kwargs, **_pypads_hook_params):
        from pypads.pypads import get_current_pads
        from padrepads.base import PyPadrePads
        pads: PyPadrePads = get_current_pads()

        if pads.cache.exists("parameter_search"):
            with pads.api.intermediate_run(experiment_id=pads.api.active_run().info.experiment_id):
                out = _pypads_env.callback(*_args, **_kwargs)
            return out
        else:
            return _pypads_env.callback(*_args, **_kwargs)
