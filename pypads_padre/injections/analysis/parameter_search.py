from pypads.app.injections.base_logger import LoggerCall, TrackedObject, LoggerOutput, OriginalExecutor
from pypads.app.injections.injection import InjectionLogger
from pypads.app.env import InjectionLoggerEnv
from pydantic import BaseModel
from pypads import logger
from pypads.importext.versioning import LibSelector
from pypads.model.logger_output import TrackedObjectModel, OutputModel
from typing import List, Optional, Type

from pypads_padre.concepts.util import validate_type


class ParameterSearchTO(TrackedObject):
    """
    Tracking object for grid search and results
    """
    class ParamSearchModel(TrackedObjectModel):
        """
        Model defining values of the parameter search entries.
        """
        catergory: str = "ParameterSearch"
        name = "GridSearchResults"
        description = "Tracked object holding the results of Sklearn CrossValidation Grid Search."

        class SearchModel(BaseModel):
            index: int = ...
            setting: dict = {}
            mean_score: float = ...
            std_score: float = ...
            ranking: int = ...

        number_of_splits: int = ...
        best_candidate: int = ...
        results: List[SearchModel] = []

        class Config:
            orm_mode = True

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.ParamSearchModel

    def __init__(self, *args, parent: LoggerOutput, **kwargs):
        super().__init__(*args, parent=parent, **kwargs)
        self.results = []

    def add_results(self, cv_results: dict):
        """
        Parse the result dict of sklearn Grid search
        """
        logger.info("Logging Grid Search resutls....")
        mean_scores = validate_type(cv_results.get('mean_test_score', []))
        std_scores = validate_type(cv_results.get('std_test_score', []))
        rankings = validate_type(cv_results.get('rank_test_score', []))
        for i, params in enumerate(cv_results.get('params', [])):
            self.results.append(self.ParamSearchModel.SearchModel(index=validate_type(i), setting=validate_type(params),
                                                                  mean_score=mean_scores[i], std_score=std_scores[i],
                                                                  ranking=rankings[i]))


class ParameterSearchILF(InjectionLogger):
    """
    Function logging the cv results of a parameter search
    """
    name = "Parameter Search Logger"
    category = "ParameterSearchLogger"

    supported_libraries = {LibSelector(name="sklearn", constraint="*", specificity=1)}

    class ParameterSearchOutput(OutputModel):
        category: str = "ParameterSearchOutput"

        gridsearch_cv: str = None

        class Config:
            orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Optional[Type[OutputModel]]:
        return cls.ParameterSearchOutput

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
        pads.cache.run_pop("parameter_search")
        from sklearn.model_selection._search import BaseSearchCV
        if isinstance(ctx, BaseSearchCV):
            gridsearch = ParameterSearchTO(parent=_logger_output)
            gridsearch.number_of_splits = ctx.n_splits_

            # Track individual decisions for all splits
            pads.cache.add("tracking_mode","multiple")

            gridsearch.best_candidate = ctx.best_index_
            gridsearch.add_results(ctx.cv_results_)
            _logger_output.gridsearch_cv = gridsearch.store()


class ParameterSearchExecutor(InjectionLogger):

    def __call_wrapped__(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _logger_call, _logger_output, _args,
                     _kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        if pads.cache.run_get("parameter_search", False):
            logger.info("Executing a parameter search under a nested run.")
            with pads.api.intermediate_run(experiment_id=pads.api.active_run().info.experiment_id, clear_cache=False,
                                           setups=False):
                _return, time = OriginalExecutor(fn=_pypads_env.callback)(*_args, **_kwargs)
            return _return, time
        else:
            return OriginalExecutor(fn=_pypads_env.callback)(*_args, **_kwargs)