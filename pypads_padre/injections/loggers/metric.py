from pypads.importext.versioning import LibSelector
from pypads.injections.loggers.metric import MetricILF, MetricTO


class MetricTorch(MetricILF):
    """
    Function logging wrapped metrics of PyTroch
    """

    supported_libraries = {LibSelector(name="torch", constraint="*", specificity=1)}

    _dependencies = {"torch"}

    def __post__(self, ctx, *args, _pypads_artifact_fallback=False, _logger_call, _logger_output, _pypads_result,
                 **kwargs):
        """
        :param ctx:
        :param args:
        :param _pypads_artifact_fallback: Write to artifact if metric can not be logged as an double value into mlflow
        :param _pypads_result:
        :param kwargs:
        :return:
        """
        result = _pypads_result
        metric = MetricTO(parent=_logger_output, as_artifact=_pypads_artifact_fallback)
        if result is not None:
            from torch import Tensor
            if isinstance(result, Tensor):
                super().__post__(ctx, *args, _pypads_artifact_fallback=_pypads_artifact_fallback,
                                 _logger_call=_logger_call, _logger_output=_logger_output,
                                 _pypads_result=result.item(), **kwargs)
            else:
                from torch.optim.optimizer import Optimizer
                if isinstance(ctx, Optimizer):
                    # Logging the gradient of the weigths after the optimizer step
                    param_groups = ctx.param_groups
                    for group in param_groups:
                        weights_by_layer = group.get('params', None)
                        if weights_by_layer and isinstance(weights_by_layer, list):
                            for layer, weights in enumerate(weights_by_layer):
                                metric.store_value(weights.grad.mean().item(),
                                                   step=_logger_call.original_call.call_id.call_number,
                                                   name="Layer_" + str(layer) + "_mean_gradient")
