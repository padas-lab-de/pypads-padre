from pypads.bindings.event_types import EventType, event_types

DEFAULT_PADRE_EVENT_TYPES = [EventType("dataset", "A function returning a dataset."),
                             EventType("predictions", "A function predicting on instance level."),
                             EventType("parameter_search", "A function denoting that a parameter search is starting"),
                             EventType("parameter_search_executor",
                                       "A function being the execution of a single parameter search run"),
                             EventType("splits", "A function providing a splitting for the dataset."),
                             EventType("hyperparameters", "A function providing hyperparameters"),
                             EventType("doc", "A function providing for critical docs for the type of experiment")]


def init_event_types():
    if not all([a.name in event_types for a in DEFAULT_PADRE_EVENT_TYPES]):
        raise Exception("There seems to be an issues with adding the anchors")


# init_event_types()
