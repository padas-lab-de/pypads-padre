from typing import List, Type, Union

from pydantic import BaseModel

from pypads import logger
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import InjectionLogger
from pypads.app.injections.tracked_object import TrackedObject, LoggerOutput
from pypads.importext.versioning import LibSelector
from pypads.model.logger_call import ContextModel
from pypads.model.logger_output import OutputModel, TrackedObjectModel
from pypads.model.models import IdReference
from pypads.utils.logging_util import data_str, data_path, add_data


class FunctionParametersTO(TrackedObject):
    """
    Tracking object class for model hyper parameters.
    """

    class FunctionParametersModel(TrackedObjectModel):
        type: str = "ModelHyperParameter"
        description = "The parameters of the experiment."
        module: str = ...
        contextmodel: ContextModel = ...
        hyper_parameters: List[IdReference] = []

    def __init__(self, *args, parent: Union[OutputModel, 'TrackedObject'], **kwargs):
        super().__init__(*args, parent=parent, **kwargs)
        self.contextmodel = self.producer.original_call.call_id.context

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.FunctionParametersModel

    def persist_parameter(self: Union['FunctionParametersTO', FunctionParametersModel], key, value, param_type=None,
                          description=None, additional_data=None):
        """
        Persist a new parameter to the tracking object.
        :param key: Name of the parameter
        :param value: Value of the parameter. This has to be convert-able to string
        :param param_type: Type of the parameter. This should store the real type of the parameter. It could be used
        to load the data in the right format from the stored string.
        :param description: A description of the parameter to be stored.
        :param additional_data: Additional data to store about the parameter.
        :return:
        """
        description = description or "Parameter named {} of context {}".format(key, self.contextmodel)
        self.hyper_parameters.append(self.store_param(key, value, param_type=param_type, description=description,
                                                      additional_data=additional_data))


class ParametersTorchILF(InjectionLogger):
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

    class ParametersTorchILFOutput(OutputModel):
        """
        Output of the logger. An output can reference multiple Tracked Objects or Values directly. In this case a own
        tracked object doesn't give a lot of benefit but enforcing a description a name and a category and could be omitted.
        """
        type: str = "ParametersTorchILF-Output"
        hyper_parameter_to: IdReference = ...

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.ParametersTorchILFOutput

    def __post__(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _logger_call,
                 _logger_output: Union['ParametersTorchILFOutput', LoggerOutput], _args, _kwargs, **kwargs):
        """
        Function logging the parameters of the current pipeline object function call.
        """

        mapping_data = _pypads_env.data

        # Get the estimator name
        module = data_str(mapping_data, "module", "@schema", "rdfs:label",
                          default=ctx.__class__.__name__)

        hyper_params = FunctionParametersTO(module=module,
                                            description=f"The parameters of estimator {module} with {ctx}.",
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
                    relevant_parameters = [{"name": k, "value": v} for k, v in defaults.items()]
                else:
                    logger.warning('Hyper Parameters extraction of optimizer {} failed'.format(str(ctx)))
            # TODO Hyper Parameters extraction for DataLoader and nn.Module custom networks
            elif isinstance(ctx, torch.utils.data.DataLoader):
                # Get all the named arguments along with default values if not given
                import inspect
                signature = inspect.signature(_pypads_env.callback)
                defaults = {k: v.default for k, v in signature.parameters.items() if
                            v.default is not inspect.Parameter.empty}
                relevant_parameters = [{"name": k, "value": v} for k, v in {**defaults, **_kwargs}.items()]
            else:
                logger.warning('Hyper Parameters extraction of {} failed'.format(str(ctx)))
        for i, param in enumerate(relevant_parameters):
            name = data_path(param, "name", default="UnknownParameter" + str(i))
            description = data_path(param, "description")
            value = data_path(param, "value")
            parameter_type = data_path(param, "parameter_type", default=str(type(value)))

            hyper_params.persist_parameter(name, value, param_type=parameter_type, description=description,
                                           additional_data=mapping_data)

        _logger_output.hyper_parameter_to = hyper_params.store()
