from typing import Type, Union, List, Dict, Optional

from pydantic import BaseModel
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import MultiInjectionLogger
from pypads.app.injections.tracked_object import TrackedObject, LoggerOutput
from pypads.app.pypads import get_current_pads
from pypads.model.logger_output import OutputModel, TrackedObjectModel
from pypads.model.models import IdReference
from pypads import logger

from pypads_padre.util import update_track, track_init

_epoch = 0

class ModelTO(TrackedObject):
    """
    Tracking object class for model hyper parameters.
    """

    class TorchModel(TrackedObjectModel):
        type: str = "TorchModel"
        description = "Information on the pytorch model used."
        Model: Optional[str] = None  # reference to the saved model artifact ?!
        weights: Dict = dict()
        gradients: Dict = dict()
        log_frequency: int = ...

    def __init__(self, *args, parent: Union[OutputModel, 'TrackedObject'], log_frequency, **kwargs):
        super().__init__(*args, parent=parent, log_frequency=log_frequency, **kwargs)

        self._hook_handles = dict()
        self._model = None

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.TorchModel

    def add_weights_tensor(self, tensor, name):
        name = name.replace('.', "_")
        hist, bins = self.tensor_stats(tensor)
        global _epoch
        if name not in self.weights:
            self.weights[name] = [{"values": hist, "bins": bins, "epoch": _epoch}]
        else:
            self.weights[name].append({"values": hist, "bins": bins, "epoch": _epoch})

    def add_grad_tensor(self, tensor, name):
        name = name.replace('.', "_")
        global _epoch
        hist, bins = self.tensor_stats(tensor)
        if name not in self.gradients:
            self.gradients[name] = [{"values": hist, "bins": bins, "epoch": _epoch}]
        else:
            self.gradients[name].append({"values": hist, "bins": bins, "epoch": _epoch})

    def tensor_stats(self, tensor, num_bins=64):
        import torch
        import numpy as np
        if isinstance(tensor, tuple) or isinstance(tensor, list):
            while (isinstance(tensor, tuple) or isinstance(tensor, list)) and (
                    isinstance(tensor[0], tuple) or isinstance(tensor[0], list)
            ):
                tensor = [item for sublist in tensor for item in sublist]
            tensor = torch.cat([t.reshape(-1) for t in tensor])

        if not hasattr(tensor, "shape"):
            cls = type(tensor)
            logger.error("Expected Tensor, not {}.{}".format(cls.__module__, cls.__name__))
            return

        # Sparse tensors have a bunch of implicit zeros. In order to histo them correctly,
        # we have to count them up and add them to the histo ourselves.
        sparse_zeros = None
        if tensor.is_sparse:
            # Have to call this on a sparse tensor before most other ops.
            tensor = tensor.cpu().coalesce().clone().detach()

            backing_values = tensor._values()
            non_zero_values = backing_values.numel()
            all_values = tensor.numel()
            sparse_zeros = all_values - non_zero_values
            tensor = backing_values

        flattened = tensor.reshape(-1)

        if not hasattr(flattened, "detach"):
            tensor = flattened.cpu().clone().numpy()
            hist, bins = np.histogram(tensor, bins=num_bins)
            hist, bins = hist.tolist(), bins.tolist()
            if len(hist) + 1 != len(bins):
                logger.error("Length of bins must be equal to the length of the histogram")
                return
            return hist, bins

        if flattened.is_cuda:
            flattened = flattened.cpu().clone().detach()

        flattened = flattened[~torch.isnan(flattened)]
        flattened = flattened[~torch.isinf(flattened)]
        if flattened.shape == torch.Size([0]):
            # Often the whole tensor is nan or inf. Just don't log it in that case.
            return

        min = flattened.min().item()
        max = flattened.max().item()

        if sparse_zeros:
            min = 0 if min > 0 else min
            max = 0 if max < 0 else max

        if min > max:
            min, max = max, min

        # compute hist
        tensor = flattened.histc(bins=num_bins, min=min, max=max)
        tensor = tensor.cpu().clone().detach()
        bins = torch.linspace(min, max, steps=num_bins + 1)

        if sparse_zeros:
            bins_np = bins.numpy()
            tensor_np = tensor.numpy()
            bin_idx = 0
            num_buckedts = len(bins_np) - 1
            for i in range(num_buckedts):
                start = bins_np[i]
                end = bins_np[i + 1]
                if (start <= 0 and end > 0) or (i == num_buckedts - 1 and end == 0):
                    bin_idx = i
                    break

            tensor_np[bin_idx] += sparse_zeros
            tensor = torch.Tensor(tensor_np)
            bins = torch.Tensor(bins_np)

        return tensor.tolist(), bins.tolist()


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
        to = output.model_to
        # TODO save the torch model and hold the reference
        output.model_to = to.store()
        logger_call.output = output.store()

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

        _pypads_tracking_freq = 5
        if pads.cache.run_exists('_pypads_tracking_freq'):
            _pypads_tracking_freq = pads.cache.run_get('_pypads_tracking_freq')

        def add_log_hooks(module):
            """
            A function that add forward and backward hooks to log module parameters and gradients.
            """
            if not isinstance(module, torch.nn.Module):
                logger.warning("Expected a Torch model (torch.nn.Module) and got {}".format(str(type(module))))
                return

            if _logger_output.model_to is None:
                model = ModelTO(parent=_logger_output, log_frequency=_pypads_tracking_freq)
                model._model = module
            else:
                model = _logger_output.model_to

            prefix = module.__class__.__name__

            # Logging parameters
            def log_weights_hook(module, input_, output, track_counter):
                if module.training:
                    if _logger_output.model_to is None:
                        model = ModelTO(parent=_logger_output, log_frequency=_pypads_tracking_freq)
                        model._model = module
                    else:
                        model = _logger_output.model_to

                    if not update_track(track_counter):
                        return
                    for name, parameter in module.named_parameters():
                        if isinstance(parameter, torch.autograd.Variable):
                            data = parameter.data
                        else:
                            data = parameter

                        model.add_weights_tensor(data.cpu(), "parameters/" + prefix + name)

            weights_counter = track_init(_pypads_tracking_freq)
            hook = module.register_forward_hook(
                lambda module, inp, out: log_weights_hook(
                    module, inp, out, weights_counter
                )
            )
            model._hook_handles["parameters/" + prefix] = hook

            # Logging gradients
            def variable_grad_hook(variable, hook_name, track_counter):
                if not isinstance(variable, torch.autograd.Variable):
                    cls = type(variable)
                    logger.warning("Expected torch.Variable, got {}.{}".format(cls.__module__, cls.__name__))
                    return
                handle = model._hook_handles.get(hook_name, None)
                if handle is not None:
                    logger.warning('A hook has already been set under name "{}"'.format(hook_name))
                    return

                def _callback(grad, track_counter):
                    if _logger_output.model_to is None:
                        model = ModelTO(parent=_logger_output, log_frequency=_pypads_tracking_freq)
                        model._model = ctx
                    else:
                        model = _logger_output.model_to

                    if not update_track(track_counter):
                        return
                    model.add_grad_tensor(grad.data, hook_name)
                    # ToDo log tensor stats

                handle = variable.register_hook(lambda grad: _callback(grad, track_counter))
                model._hook_handles[hook_name] = handle

            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    grad_counter = track_init(_pypads_tracking_freq)
                    variable_grad_hook(parameter, "gradients/" + prefix + name, grad_counter)

            _logger_output.model_to = model

        add_log_hooks(ctx)
