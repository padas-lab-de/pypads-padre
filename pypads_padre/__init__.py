__version__ = '0.2.2'

from pypads_padre.app.plugin import configure_plugin


# Entrypoint for the plugin TODO allow to disable this we could also call a defined entrypoint from pypads and decide
def activate():
    configure_plugin()
