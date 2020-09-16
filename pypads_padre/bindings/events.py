from pypads_padre.injections.analysis.doc_parsing import Doc
from pypads_padre.injections.analysis.parameter_search import ParameterSearch, ParameterSearchExecutor
from pypads_padre.injections.loggers.data_splitting import SplitILF, SplitILFTorch
from pypads_padre.injections.loggers.dataset import DatasetILF
from pypads_padre.injections.loggers.decision_tracking import SingleInstanceILF, Decisions_keras, Decisions_sklearn, \
    Decisions_torch
from pypads_padre.injections.loggers.hyperparameters import HyperParameters
from pypads_padre.injections.loggers.metric import MetricTorch

# Extended mappings. We allow to log parameters, output or input, datasets
DEFAULT_PADRE_LOGGING_FNS = {
    "dataset": DatasetILF(),
    "predictions": [SingleInstanceILF(), Decisions_sklearn(), Decisions_torch(), Decisions_keras()],
    "parameter_search": ParameterSearch(),
    "parameter_search_executor": ParameterSearchExecutor(),
    "splits": [SplitILF(), SplitILFTorch()],
    "hyperparameters": HyperParameters(),
    "doc": Doc(),
    "metric": [MetricTorch()]
}
