from types import GeneratorType
from typing import Tuple, Iterable

from pypads import logger
from pypads.app.injections.base_logger import LoggingFunction, OriginalExecutor
from pypads.importext.mappings import LibSelector
from pypads.injections.analysis.call_tracker import LoggingEnv
from pypads.injections.analysis.time_keeper import add_run_time, TimingDefined


def split_output_inv(result, fn=None):
    # function that looks into the output of the custom splitter
    split_info = dict()

    # Flag to check whether the outputs of the splitter are indices (one dimensional Iterable)
    indices = True
    if isinstance(result, Tuple):
        n_output = len(result)
        for a in result:
            if isinstance(a, Iterable):
                for row in a:
                    if isinstance(row, Iterable):
                        indices = False
                        break

        if n_output > 3:
            if indices:
                logger.warning(
                    'The splitter function return values are ambiguous (more than train/test/validation splitting).'
                    'Decision tracking might be inaccurate')
                split_info.update({'set_{}'.format(i): a for i, a in enumerate(result)})
                split_info.update({"track_decisions": False})
            else:
                logger.warning("The output of the splitter is not indices, Decision tracking might be inaccurate.")
                if "sklearn" in fn.__module__:
                    split_info.update({'Xtrain': result[0], 'Xtest': result[1], 'ytrain': result[2],
                                       'ytest': result[3]})
                    split_info.update({"track_decisions": True})
                else:
                    split_info.update({'output_{}'.format(i): a for i, a in enumerate(result)})
                    split_info.update({"track_decisions": False})
        else:
            if indices:
                names = ['train', 'test', 'val']
                i = 0
                while i < n_output:
                    split_info[names[i]] = result[i]
                    i += 1
                split_info.update({"track_decisions": True})
            else:
                logger.warning("The output of the splitter is not indices, Decision tracking might be inaccurate.")
                split_info.update({'output_{}'.format(i): a for i, a in enumerate(result)})
                split_info.update({"track_decisions": False})
    else:
        logger.warning("The splitter has a single output. Decision tracking might be inaccurate.")
        split_info.update({'output_0': result})
        split_info.update({"track_decisions": True})
    return split_info


class SplitsTracker(LoggingFunction):
    """
    Function that tracks data splits
    """

    def __call_wrapped__(self, ctx, *args, _pypads_env: LoggingEnv, _args, _kwargs, **_pypads_hook_params):
        """

        :param ctx:
        :param args:
        :param _pypads_result:
        :param kwargs:
        :return:
        """

        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        _return, time = OriginalExecutor(fn=_pypads_env.callback)(*_args, **_kwargs)
        try:
            add_run_time(None, str(_pypads_env.call), time)
        except TimingDefined as e:
            pass

        if isinstance(_return, GeneratorType):
            def generator():
                num = -1
                for r in _return:
                    num += 1
                    pads.cache.run_add("current_split", num)
                    split_info = split_output_inv(r, fn=_pypads_env.callback)
                    pads.cache.run_add(num, {"split_info": split_info})
                    yield r
        else:
            def generator():
                split_info = split_output_inv(_return, fn=_pypads_env.callback)
                pads.cache.run_add("current_split", 0)
                pads.cache.run_add(0, {"split_info": split_info})

                return _return

        return generator()


class SplitsTrackerTorch(LoggingFunction):

    def supported_libraries(self):
        return {LibSelector("torch", "*", specificity=1)}

    def _handle_error(self, *args, ctx, _pypads_env, error, **kwargs):
        if isinstance(error, StopIteration):
            logger.warning("Ignoring recovery of this StopIteration error: {}".format(error))
            original = _pypads_env.call.call_id.context.original(_pypads_env.callback)
            return original(ctx, *args, **kwargs)
        else:
            super()._handle_error(*args, ctx, _pypads_env, error, **kwargs)

    def __post__(self, ctx, *args, _pypads_env, _pypads_pre_return, _pypads_result, _args, _kwargs, **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        # Dataloader splits
        train = True
        if hasattr(ctx, "_dataset"):
            train = ctx._dataset.train

        if not train:
            curr = pads.cache.run_get("current_split", None)
            if curr is not None:
                curr += 1
            else:
                curr = 0
            pads.cache.run_add("current_split", curr)
            pads.cache.run_add(curr, {"split_info": {"test": _pypads_result}})
