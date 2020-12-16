from pypads import logger

from test.base_test import BaseTest
from test.test_torch.util import torch_simple_example, torch_3d_mnist_example


# noinspection PyMethodMayBeStatic
class PyPadsTorchTest(BaseTest):

    def test_torch_sequential_class(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(autostart="MNIST-Torch", setup_fns=[])
        try:
            import timeit
            t = timeit.Timer(torch_simple_example)
            print(t.timeit(1))
        except RuntimeError as e:
            logger.warning("Torch bug on re-import: {}".format(str(e)))
        # --------------------------- asserts ---------------------------
        # TODO Add asserts
        # !-------------------------- asserts ---------------------------
        tracker.api.end_run()

    def test_3d_mnist(self):
        # --------------------------- setup of the tracking ---------------------------
        try:
            import timeit
            t = timeit.Timer(torch_3d_mnist_example)
            print(t.timeit(1))
        except RuntimeError as e:
            logger.warning("Torch bug on re-import: {}".format(str(e)))
        # --------------------------- asserts ---------------------------
        # TODO Add asserts
        # !-------------------------- asserts ---------------------------
