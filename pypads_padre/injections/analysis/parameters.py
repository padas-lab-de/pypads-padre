from typing import Union, Iterable

from pypads import logger
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.tracked_object import LoggerOutput
from pypads.injections.analysis.parameters import ParametersILF, FunctionParametersTO
from pypads.utils.logging_util import data_str, data_path, add_data


def _get_relevant_parameters(model):
    import torch
    import inspect
    layers = dict()
    n = 0
    total_params = 0
    trainable_params = 0
    for i, m in enumerate(model.modules()):
        if i == 0:
            continue
        else:
            if isinstance(m, torch.nn.Sequential):
                continue
            else:
                n += 1
                trainable = False
                params = 0
                signature = inspect.signature(m.__init__)
                keys = [k for k in signature.parameters.keys()]
                # extracting information from the layer
                for k in keys:
                    if k in m.__dict__:
                        p = m.__dict__.get(k)
                        if isinstance(p, Iterable):
                            p = str(p)
                        layers["{}.{}".format(m.__class__.__name__, k)] = p
                if hasattr(m, "weight") and hasattr(m.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(m.weight.size()))).item()
                    trainable = m.weight.requires_grad
                if hasattr(m, "bias") and hasattr(m.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(m.bias.size()))).item()
                layers["{}.{}".format(m.__class__.__name__, "parameters")] = params
                total_params += params
                if trainable:
                    trainable_params += params

    layers["{}.Trainable_parameters".format(model.__class__.__name__)] = trainable_params
    layers["{}.Total_parameters".format(model.__class__.__name__)] = total_params
    layers["{}.Number_of_layers".format(model.__class__.__name__)] = n

    return layers


# noinspection PyMethodMayBeStatic, DuplicatedCode
class ParametersTorchILF(ParametersILF):
    """
    Function logging the hyper parameters of the current pipeline object. This stores parameters as pypads parameters.
    Mapping files should give data in the format of:

        Hook:
            Hook this logger to function calls on which an estimator parameter setting should be extracted. For sklearn
            this is the fit, fit_predict etc. function. Generally one could log parameters on initialisation of an
            estimator too but this would't be able to track changes to the parameter settings done in between inti
            and fitting.
        Mapping_File:

    """

    name = "Torch Parameter Logger"
    type: str = "TorchParameterLogger"

    _dependencies = {"torch"}

    def __post__(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _logger_call,
                 _logger_output: Union['ParametersILF.ParametersILFOutput', LoggerOutput], _args, _kwargs, **kwargs):
        """
        Function logging the parameters of the current pipeline object function call.
        """

        mapping_data = _pypads_env.data

        # Get the estimator name
        module = data_str(mapping_data, "module", "@schema", "rdfs:label",
                          default=ctx.__class__.__name__)

        hyper_params = FunctionParametersTO(estimator=module,
                                            description=f"The parameters of model {module} with {ctx}.",
                                            parent=_logger_output)

        # List of parameters to extract. Either provided by a mapping file or by get_params function or by _kwargs
        relevant_parameters = []

        if data_path(_pypads_env.data, "module", "parameters",
                     warning="No parameters are defined on the mapping file for " + str(
                         ctx.__class__) + ". Trying to log parameters without schema definition programmatically."):
            relevant_parameters = []
            for parameter_type, parameters in data_path(mapping_data, "module", "parameters", default={}).items():
                for parameter in parameters:
                    parameter = data_path(parameter, "@schema")
                    key = data_path(parameter, "padre:path")
                    name = data_path(parameter, "rdfs:label")

                    param_dict = {"name": name,
                                  "description": data_path(parameter, "rdfs:comment"),
                                  "parameter_type": data_path(parameter, "padre:value_type")}

                    if hasattr(ctx, key):
                        value = getattr(ctx, key)
                    else:
                        _kwargs = getattr(kwargs, "_kwargs")
                        if hasattr(_kwargs, key):
                            value = getattr(_kwargs, key)
                        else:
                            logger.warning(f"Couldn't extract value of in schema defined parameter {parameter}.")
                            continue
                    param_dict["value"] = value
                    add_data(mapping_data, "is_a", value=data_path(parameter, "@id"))
                    relevant_parameters.append(param_dict)

        else:
            import torch
            if isinstance(ctx, torch.optim.Optimizer):
                defaults = getattr(ctx, "defaults")
                if defaults is not None:

                    # Extracting hyperparameters via defaults dict (valid for torch optimizers)
                    relevant_parameters = [{"name": "{}.{}".format(ctx.__class__.__name__, k), "value": v} for k, v in
                                           defaults.items()]
                else:
                    logger.warning('Hyper Parameters extraction of optimizer {} failed'.format(str(ctx)))
            elif isinstance(ctx, torch.utils.data.DataLoader):
                # Get all the named arguments along with default values if not given
                import inspect
                signature = inspect.signature(_pypads_env.callback)
                defaults = {k: v.default for k, v in signature.parameters.items() if
                            v.default is not inspect.Parameter.empty}
                relevant_parameters = [{"name": "{}.{}".format(ctx.__class__.__name__, k), "value": v} for k, v in
                                       {**defaults, **_kwargs}.items()]
            elif isinstance(ctx, torch.nn.Module):
                params = _get_relevant_parameters(ctx)
                relevant_parameters = [{"name": k, "value": v} for k,v in params.items()]
            else:
                logger.warning('Hyper Parameters extraction of {} failed'.format(str(ctx)))
        for i, param in enumerate(relevant_parameters):
            name = data_path(param, "name", default="UnknownParameter" + str(i))
            description = data_path(param, "description")
            value = data_path(param, "value")
            parameter_type = data_path(param, "parameter_type", default=str(type(value)))

            hyper_params.persist_parameter(name, str(value), param_type=parameter_type, description=description,
                                           additional_data=mapping_data)

        _logger_output.hyper_parameter_to = hyper_params.store()
