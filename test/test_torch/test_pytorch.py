import os

from test.base_test import BaseTest, _get_mapping, TEST_FOLDER
from test.test_torch.test_util import torch_simple_example

# torch_padre = _get_mapping(os.path.join(os.path.dirname(__file__), "torch_1_4_0.yml"))
#
# torchvision_padre = _get_mapping(os.path.join(os.path.dirname(__file__), "torchvision_0_5_0.yml"))


# noinspection PyMethodMayBeStatic
class PyPadsTorchTest(BaseTest):

    def test_torch_sequential_class(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(autostart="MNIST-Torch", setup_fns=[])

        import timeit
        t = timeit.Timer(torch_simple_example)
        print(t.timeit(1))
        # --------------------------- asserts ---------------------------
        # TODO Add asserts
        # !-------------------------- asserts ---------------------------

    def test_3d_mnist(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(autostart="MNIST-Torch", setup_fns=[])

        import timeit
        t = timeit.Timer(torch_simple_example)
        print(t.timeit(1))
        # --------------------------- asserts ---------------------------
        # TODO Add asserts
        # !-------------------------- asserts ---------------------------