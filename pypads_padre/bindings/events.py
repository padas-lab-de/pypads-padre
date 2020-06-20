from pypads_padre.injections.analysis.doc_parsing import Doc
from pypads_padre.injections.analysis.parameter_search import ParameterSearch, ParameterSearchExecutor
from pypads_padre.injections.loggers.data_splitting import SplitsTracker
from pypads_padre.injections.loggers.dataset import Dataset
from pypads_padre.injections.loggers.decision_tracking import Decisions, Decisions_keras, Decisions_sklearn, \
    Decisions_torch
from pypads_padre.injections.loggers.hyperparameters import HyperParameters
from pypads_padre.injections.loggers.metric import MetricTorch


# Extended mappings. We allow to log parameters, output or input, datasets
DEFAULT_PADRE_LOGGING_FNS = {
    "dataset": Dataset(),
    "predictions": Decisions(),
    "parameter_search": ParameterSearch(),
    "parameter_search_executor": ParameterSearchExecutor(),
    "splits": SplitsTracker(),
    "hyperparameters": HyperParameters(),
    "doc": Doc(),
    ("predictions", "keras"): Decisions_keras(),
    ("predictions", "sklearn"): Decisions_sklearn(),
    ("predictions", "torch"): Decisions_torch(),
    ("metric", "torch"): MetricTorch()
}
