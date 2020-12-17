__version__ = '0.4.2'

from pypads import logger


# Entrypoint for the plugin TODO allow to disable this we could also call a defined entrypoint from pypads and decide
def activate(pypads, *args, **kwargs):
    from pypads_padre.app.plugin import configure_plugin
    logger.info("Trying to configure padre plugin for pypads...")
    configure_plugin(pypads, *args, **kwargs)
    logger.info("Finished configuring padre plugin for pypads!")
