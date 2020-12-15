from test.base_test import BaseTest
from test.test_torch.util import torch_simple_example, torch_3d_mnist_example


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

        import timeit
        t = timeit.Timer(torch_3d_mnist_example)
        print(t.timeit(1))
        # --------------------------- asserts ---------------------------
        # TODO Add asserts
        # !-------------------------- asserts ---------------------------
