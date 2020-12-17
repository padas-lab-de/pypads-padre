from pypads.bindings.anchors import Anchor, anchors

DEFAULT_ANCHORS = [Anchor("pypads_dataset", "Used if a dataset is being detected"),
                   Anchor("pypads_split", "Used when splitting the dataset"),
                   Anchor("pypads_params", "TODO"),
                   Anchor("pypads_param_search", "TODO"),
                   Anchor("pypads_param_search_exec", "TODO"),
                   Anchor("pypads_grad", "TODO")]


def init_anchors():
    if not all([a.name in anchors for a in DEFAULT_ANCHORS]):
        raise Exception("There seems to be an issues with adding the anchors")


# init_anchors()
