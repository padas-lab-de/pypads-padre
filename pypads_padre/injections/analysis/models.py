from typing import Type, Union

from pydantic import BaseModel
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import MultiInjectionLogger
from pypads.app.injections.tracked_object import TrackedObject, LoggerOutput
from pypads.app.pypads import get_current_pads
from pypads.model.logger_output import OutputModel, TrackedObjectModel
from pypads.model.models import IdReference


class ModelTO(TrackedObject):
    """
    Tracking object class for model hyper parameters.
    """

    class TorchModel(TrackedObjectModel):
        type: str = "TorchModel"
        description = "Information on the pytorch model used."
        Model: str = ...

    def __init__(self, *args, parent: Union[OutputModel, 'TrackedObject'], **kwargs):
        super().__init__(*args, parent=parent, **kwargs)

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.TorchModel


class TorchModelILF(MultiInjectionLogger):
    """
    Function logging everything we can about a pytorch model. This stores information on layers, weights, gradients, etc.


    """

    name = "Torch Model Logger"
    type: str = "TorchModelLogger"

    _dependencies = {"torch"}

    class TorchModelILFOutput(OutputModel):
        """
        Output of the logger. An output can reference multiple Tracked Objects or Values directly. In this case a own
        tracked object doesn't give a lot of benefit but enforcing a description a name and a category and could be omitted.
        """
        type: str = "TorchModelILF-Output"
        model_to: IdReference = ...

    @staticmethod
    def finalize_output(pads, logger_call, output, *args, **kwargs):
        pass

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.TorchModelILFOutput

    def __post__(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _logger_call,
                 _logger_output: Union['TorchModelILFOutput', LoggerOutput], _args, _kwargs, **kwargs):
        """
        Function logging information about the logger
        """
        import torch

        pads = get_current_pads()

        mapping_data = _pypads_env.data

        # Initialize epoch value
        epoch = 0
        if not pads.cache.run_exists('epoch'):
            pads.cache.run_add('epoch', epoch)

        _pypads_avg_window = 5
        if pads.cache.run_exists('pypads_averaging_window'):
            _pypads_avg_window = pads.cache.run_get('pypads_averaging_window')

        hooks = []

        def register_forward_hook(module):
            def hook(module, input, output):
                #TODO
                pass
            if (
                    not isinstance(module, torch.nn.Sequential)
                    and not isinstance(module, torch.nn.ModuleList)
            ):
                hooks.append(module.register_forward_hook(hook))

        # Todo registering hooks and extracting information on the torch model.

        def register_backward_hook(module):
            def hook(module, grad_in, grad_out):
                #TODO
                pass
            if (
                    not isinstance(module, torch.nn.Sequential)
                    and not isinstance(module, torch.nn.ModuleList)
            ):
                hooks.append(module.register_backward_hook(hook))


        def cleanup_hooks(hooks):
            for hook in hooks:
                hook.remove()

        pass
