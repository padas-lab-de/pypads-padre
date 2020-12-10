from pypads.injections.loggers.debug import Log
from pypads.utils.util import dict_merge
from pypads.app.injections.injection import InjectionLogger
from copy import deepcopy

import os
import numpy as np
from typing import Any
from collections import OrderedDict
from pypads.importext.versioning import LibSelector

from pypads.app.base import PyPads


def main():
    test_folder = os.path.join(os.path.abspath("~"), ".pypads-test_" + str(os.getpid()))
    # torch_events = {
    #     "step": [Log(), PyTorchUpdateDebugger()]
    # }
    torch_hooks = {
        "step": {"on": ["pypads_step"]}
    }
    config = {'mongo_db': False}

    from pypads.bindings.hooks import DEFAULT_HOOK_MAPPING
    tracker = PyPads(uri=test_folder, config=config, hooks=dict_merge(DEFAULT_HOOK_MAPPING, torch_hooks),
                     autostart=True)

    import torch
    from torch import nn

    class Model(nn.Module):
        def _forward_unimplemented(self, *input: Any) -> None:
            pass

        def __init__(self, feature_shape, leaky_value=0.05, shape=[30, 50, 1], bias=True):
            # Model can have as many layers as the user wishes
            super(Model, self).__init__()
            layers = [feature_shape] + shape
            layer_list = OrderedDict()
            for idx in range(1, len(layers) - 1):
                layer = None
                layer = nn.Linear(layers[idx - 1], layers[idx], bias=bias)

                torch.nn.init.xavier_uniform_(layer.weight)

                layer_list[str(idx)] = layer

                layer_list['Relu' + str(idx)] = nn.Hardswish()

            # layer_list['Relu'] = nn.ReLU()
            layer_list['final'] = nn.Linear(layers[-2], layers[-1], bias=bias)
            self.model = nn.Sequential(layer_list)

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
            return validity

    @tracker.decorators.dataset
    def load_dataset():
        """
        This function loads the dataset for the experiment
        :return: features and targets
        """
        x = np.random.random(size=(100, 3))
        y = np.random.randint(0, 3, 100)
        y = y.astype(dtype=float)
        y = np.reshape(y, newshape=(y.shape[0], 1))

    x, y = load_dataset()

    y = torch.from_numpy(y)
    x = torch.stack([torch.tensor(i[:-1]) for i in x])

    num_features = x.shape[1]
    neural_network_shape = [100, 1]
    model = Model(feature_shape=num_features, shape=neural_network_shape, bias=True)
    model = model.double()

    optimizer = torch.optim.Adadelta(model.parameters())
    loss_ = torch.nn.MSELoss()

    optimizer.zero_grad()

    # Compute loss
    forward = model.forward(x)
    _loss = loss_(forward, y)

    # Do backpropagation and update weights
    _loss.backward()
    optimizer.step()
    print('Run Completed')
    return


if __name__ == '__main__':
    main()
