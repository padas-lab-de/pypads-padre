from pypads.app.injections.base_logger import LoggerCall, TrackedObject, LoggerOutput
from pypads.app.injections.injection import InjectionLogger
from pypads.app.env import InjectionLoggerEnv
from pydantic import BaseModel

from pypads.importext.versioning import LibSelector
from pypads.model.logger_output import TrackedObjectModel, OutputModel
from typing import List


class ParameterSearchTO(TrackedObject):
    """
    Tracking object for grid search and results
    """

    class ParamSearchModel(TrackedObjectModel):
        catergory: str = "ParameterSearch"

        class SearchModel(BaseModel):
            index: int = ...
            setting: dict = {}
            mean_score: float = ...
            std_score: float = ...
            ranking: int = ...

        number_of_splits: int = ...
        results: List[SearchModel] = []

        class Config:
            orm_mode = True

    def __init__(self, *args, part_of: LoggerOutput, **kwargs):
        super().__init__(*args, part_of=part_of, **kwargs)

    def add_results(self, cv_results: dict):
        """
        Parse the result dict of sklearn Grid search
        """
        mean_scores = cv_results.get('mean_test_score',[])
        std_scores = cv_results.get('std_test_score', [])
        rankings = cv_results.get('rank_test_score', [])
        for i, params in enumerate(cv_results.get('params', [])):
            #TODO
            pass


class ParameterSearchILF(InjectionLogger):
    """
    Function logging the cv results of a parameter search
    """
    name = "ParameterSearchILF"
    category = f"ParameterSearchLogger"

    supported_libraries = {LibSelector(name="sklearn", constraint="*", specificity=1)}

    class ParameterSearchOutput(OutputModel):
        category: str = "ParameterSearchOutput"

        gridsearch_cv: str = ...

        class Config:
            orm_mode = True

    def __pre__(self, ctx, *args, _pypads_write_format=None, _logger_call: LoggerCall, _logger_output, _args, _kwargs,
                **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        pads.cache.run_add("parameter_search", True)

    def __post__(self, ctx, *args, _logger_call, _pypads_pre_return,
                 _pypads_result, _logger_output, _args, _kwargs,
                 **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        pads.cache.pop("parameter_search")
        from sklearn.model_selection._search import BaseSearchCV
        if isinstance(ctx, BaseSearchCV):
            gridsearch = ParameterSearchTO(part_of=_logger_output)
            gridsearch.number_of_splits = ctx.n_splits_
            gridsearch.add_results(ctx.cv_results_)
            gridsearch.store(_logger_output, "gridsearch_cv")


class ParameterSearchExecutor(InjectionLogger):

    def call_wrapped(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _logger_call, _logger_output, _args,
                     _kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        if pads.cache.run_get("parameter_search", False):
            with pads.api.intermediate_run(experiment_id=pads.api.active_run().info.experiment_id, setups=False):
                out = _pypads_env.callback(*_args, **_kwargs)
            return out
        else:
            return _pypads_env.callback(*_args, **_kwargs)
