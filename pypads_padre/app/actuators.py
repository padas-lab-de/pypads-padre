from pypads.app.actuators import IActuators, actuator

from pypads_padre.concepts.splitter import default_split


class PadrePadsActuators(IActuators):
    def __init__(self):
        super().__init__()

    @property
    def pypads(self):
        from pypads.app.pypads import get_current_pads
        return get_current_pads()

    @actuator
    def set_random_seed(self, seed=None):
        from pypads_padre.functions.management.randomness import set_random_seed
        # Set seed if needed
        if seed is None:
            import random
            # import sys
            # seed = random.randrange(sys.maxsize)
            # Numpy only allows for a max value of 2**32 - 1
            seed = random.randrange(2 ** 32 - 1)
        self.pypads.cache.run_add('seed', seed)
        set_random_seed(seed)

    # noinspection PyMethodMayBeStatic
    @actuator
    def default_splitter(self, data, **kwargs):
        ctx = _create_ctx(self.pypads.cache.run_cache().cache)
        ctx.update({"data": data})
        return default_split(ctx, **kwargs)


def _create_ctx(cache):
    ctx = dict()
    if "data" in cache.keys():
        ctx["data"] = cache.get("data")
    if "shape" in cache.keys():
        ctx["shape"] = cache.get("shape")
    if "targets" in cache.keys():
        ctx["targets"] = cache.get("targets")
    return ctx
