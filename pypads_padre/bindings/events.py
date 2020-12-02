from pypads_padre.injections.analysis.models import TorchModelILF
from pypads_padre.injections.analysis.doc_parsing import DocExtractionILF
from pypads_padre.injections.analysis.parameter_search import ParameterSearchILF, ParameterSearchExecutor
from pypads_padre.injections.analysis.parameters import ParametersTorchILF
from pypads_padre.injections.loggers.data_splitting import SplitILF, SplitILFTorch
from pypads_padre.injections.loggers.dataset import DatasetILF
from pypads_padre.injections.loggers.decision_tracking import SingleInstanceILF, DecisionsKerasILF, DecisionsSklearnILF, \
    DecisionsTorchILF
from pypads_padre.injections.loggers.estimator import EstimatorILF
from pypads_padre.injections.loggers.metric import MetricTorch

# Extended mappings. We allow to log parameters, output or input, datasets
DEFAULT_PADRE_LOGGING_FNS = {
    "dataset": DatasetILF(),
    "hyperparameters": ParametersTorchILF(),
    # "model": TorchModelILF(),
    "predictions": [DecisionsSklearnILF(), DecisionsTorchILF(), DecisionsKerasILF(), SingleInstanceILF()],
    # "parameter_search": ParameterSearchILF(),
    # "parameter_search_executor": ParameterSearchExecutor(),
    "splits": [SplitILF(),SplitILFTorch()],
    # "doc": DocExtractionILF(),
    "metric": [MetricTorch()],
    "estimator": EstimatorILF()
}
