from pypads.app.actuators import IActuators, actuator

from pypads_padre.concepts.splitter import default_split


class PadrePadsActuators(IActuators):
    def __init__(self):
        super().__init__()

    @property
    def pypads(self):
        from pypads.app.pypads import get_current_pads
        return get_current_pads()

    # noinspection PyMethodMayBeStatic
    @actuator
    def default_splitter(self, X, y=None, **kwargs):
        return self.pypads.api.track_splits(default_split)(X, y=y, **kwargs)

