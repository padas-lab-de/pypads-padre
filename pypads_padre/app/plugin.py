import glob
import os

from pypads.app import base
from pypads.bindings import events
from pypads.bindings import hooks
from pypads.importext import mappings
from pypads.utils.util import dict_merge_caches

from pypads_padre.app.actuators import PadrePadsActuators
from pypads_padre.app.api import PadrePadsApi
from pypads_padre.app.decorators import PadrePadsDecorators
from pypads_padre.app.validators import PadrePadsValidators
from pypads_padre.bindings.anchors import init_anchors
from pypads_padre.bindings.event_types import init_event_types
from pypads_padre.bindings.events import DEFAULT_PADRE_LOGGING_FNS

# --- Pypads App ---
from pypads_padre.bindings.hooks import DEFAULT_PADRE_HOOK_MAPPING

DEFAULT_PADRE_SETUP_FNS = {}

# Extended config.
# Pypads mapping files shouldn't interact directly with the logging functions,
# but define events on which different logging functions can listen.
# This config defines such a listening structure.
# {"recursive": track functions recursively. Otherwise check the callstack to only track the top level function.}
DEFAULT_PADRE_CONFIG = {}


def configure_plugin():
    """
    This function can be used to configure the plugin. It should be called at least once to allow for the usage of the
    plugin. Multiple executions should be possible.
    :return:
    """
    actuators = PadrePadsActuators()
    validators = PadrePadsValidators()
    decorators = PadrePadsDecorators()
    api = PadrePadsApi()

    mappings.default_mapping_file_paths.extend(
        glob.glob(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bindings",
                                               "resources", "mapping", "**.json"))))
    base.DEFAULT_SETUP_FNS = dict_merge_caches(base.DEFAULT_SETUP_FNS, DEFAULT_PADRE_SETUP_FNS)
    base.DEFAULT_CONFIG = dict_merge_caches(base.DEFAULT_CONFIG, DEFAULT_PADRE_CONFIG)
    events.DEFAULT_LOGGING_FNS = dict_merge_caches(events.DEFAULT_LOGGING_FNS, DEFAULT_PADRE_LOGGING_FNS)
    hooks.DEFAULT_HOOK_MAPPING = dict_merge_caches(hooks.DEFAULT_HOOK_MAPPING, DEFAULT_PADRE_HOOK_MAPPING)
    init_event_types()
    init_anchors()

    print("Trying to configure padre plugin for pypads...")
    return actuators, validators, decorators, api
