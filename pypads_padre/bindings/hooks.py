import sys

DEFAULT_PADRE_HOOK_MAPPING = {
    "dataset": {"on": ["pypads_dataset"]},
    "predictions": {"on": ["pypads_predict"]},
    "model": {"on": ["pypads_model"]},
    "splits": {"on": ["pypads_split"]},
    "hyperparameters": {"on": ["pypads_params"]},
    "parameter_search": {"on": ["pypads_param_search"], "order": sys.maxsize - 1},
    "parameter_search_executor": {"on": ["pypads_param_search_exec"], "order": sys.maxsize - 2},
    # "doc": {"on": ["pypads_init", "pypads_dataset", "pypads_fit", "pypads_transform", "pypads_predict"]},
    "metric": {"on": ["pypads_metric"], "with": {"artifact_fallback": True}},
    "estimator": {"on": ["pypads_estimator"]}
}
