import sys

DEFAULT_PADRE_HOOK_MAPPING = {
    "input": {"with": {"no_intermediate": True}},
    "output": {"with": {"no_intermediate": True}},
    "hardware": {"with": {"no_intermediate": True}},
    "dataset": {"on": ["pypads_dataset"]},
    "predictions": {"on": ["pypads_predict"]},
    "splits": {"on": ["pypads_split"]},
    "hyperparameters": {"on": ["pypads_params"]},
    "parameter_search": {"on": ["pypads_param_search"], "order": sys.maxsize - 1},
    "parameter_search_executor": {"on": ["pypads_param_search_exec"], "order": sys.maxsize - 2},
    "doc": {"on": ["pypads_init", "pypads_dataset", "pypads_fit", "pypads_transform", "pypads_predict"]},
    "metric": {"on": ["pypads_metric", "pypads_grad"], "with": {"artifact_fallback": True}}
}