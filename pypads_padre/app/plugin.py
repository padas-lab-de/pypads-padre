import glob
import os

from pypads.app import base
from pypads.app.base import PyPads
from pypads.bindings import events
from pypads.bindings import hooks
from pypads.importext import mappings
from pypads.utils.util import dict_merge_caches, dict_merge

from pypads_padre.app.actuators import PadrePadsActuators
from pypads_padre.app.api import PadrePadsApi
from pypads_padre.app.backends.repository import DatasetRepository, EstimatorRepository
from pypads_padre.app.decorators import PadrePadsDecorators
from pypads_padre.app.results import PadrePadsResults
from pypads_padre.app.validators import PadrePadsValidators
from pypads_padre.bindings.anchors import init_anchors
from pypads_padre.bindings.event_types import init_event_types
from pypads_padre.bindings.events import DEFAULT_PADRE_LOGGING_FNS

# --- Pypads App ---
from pypads_padre.bindings.hooks import DEFAULT_PADRE_HOOK_MAPPING

DEFAULT_PADRE_SETUP_FNS = set()

# Extended config.
# Pypads mapping files shouldn't interact directly with the logging functions,
# but define events on which different logging functions can listen.
# This config defines such a listening structure.
# {"recursive": track functions recursively. Otherwise check the callstack to only track the top level function.}
DEFAULT_PADRE_CONFIG = {
    "use_pypads_default_mappings": False
}


def configure_plugin(pypads,*args,**kwargs):
    """
    This function can be used to configure the plugin. It should be called at least once to allow for the usage of the
    plugin. Multiple executions should be possible.
    :return:
    """
    actuators = PadrePadsActuators()
    validators = PadrePadsValidators()
    decorators = PadrePadsDecorators()
    api = PadrePadsApi()
    results = PadrePadsResults()

    if DEFAULT_PADRE_CONFIG.get("use_pypads_default_mappings", True):
        mappings.default_mapping_file_paths.extend(
            glob.glob(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bindings",
                                                   "resources", "mapping", "**.yml"))))
    else:
        mappings.default_mapping_file_paths = glob.glob(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bindings",
                                         "resources", "mapping", "**.yml")))
    base.DEFAULT_SETUP_FNS = base.DEFAULT_SETUP_FNS | DEFAULT_PADRE_SETUP_FNS
    base.DEFAULT_CONFIG = dict_merge(base.DEFAULT_CONFIG, DEFAULT_PADRE_CONFIG, str_to_set=True)
    events.DEFAULT_LOGGING_FNS = dict_merge(events.DEFAULT_LOGGING_FNS, DEFAULT_PADRE_LOGGING_FNS, str_to_set=True)
    hooks.DEFAULT_HOOK_MAPPING = dict_merge(hooks.DEFAULT_HOOK_MAPPING, DEFAULT_PADRE_HOOK_MAPPING, str_to_set=True)
    init_event_types()
    init_anchors()

    def add_repositories(instance):
        setattr(instance, "_dataset_repository", DatasetRepository())
        setattr(instance, "_estimator_repository", EstimatorRepository())

        PyPads.dataset_repository = property(lambda self: self._dataset_repository)
        PyPads.estimator_repository = property(lambda self: self._estimator_repository)

    pypads.add_instance_modifier(add_repositories)

    print("Trying to configure padre plugin for pypads...")
    return actuators, validators, decorators, api, results
