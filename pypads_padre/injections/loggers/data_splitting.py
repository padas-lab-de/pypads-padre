import uuid
from types import GeneratorType
from typing import Tuple, List, Type, Dict

from pydantic import BaseModel
from pypads import logger
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.base_logger import TrackedObject
from pypads.app.injections.injection import OriginalExecutor, MultiInjectionLogger
from pypads.importext.mappings import LibSelector
from pypads.model.logger_output import TrackedObjectModel, OutputModel

from pypads_padre.concepts.util import _tolist


def splitter_output(result, fn):
    # check if the output of the splitter is a tuple of indices
    try:
        if isinstance(result, Tuple):
            if "sklearn" in fn.__module__:
                return result[0].tolist(), result[1].tolist(), None, None
            elif "default_splitter" in fn.__name__:
                return result[1], result[2], result[3], result[0]
            else:
                if len(result) < 4:
                    return _tolist(result[0]), _tolist(result[1]), _tolist(result[3]), None
        else:
            if "torch" in fn.__module__:
                if hasattr(fn, "_dataset"):
                    if fn._dataset.train:
                        return result.tolist(), None, None, None
                    else:
                        return None, result.tolist(), None, None
                return result.tolist(), None, None, None
            else:
                return None, None, None, None
    except Exception as e:
        logger.warning("Split tracking ommitted due to exception {}".format(str(e)))
        return None, None, None, None


class SplitTO(TrackedObject):
    """
    Tracking Object class for splits of your tracked dataset. Splits are defined
    """

    name = "Splits"

    class SplitModel(TrackedObjectModel):
        category: str = "Split"

        class Split(BaseModel):
            train_set: List = []
            test_set: List = []
            validation_set: List = []

            class Config:
                orm_mode = True
                arbitrary_types_allowed = True

        # splits: List[Split] = []
        splits: Dict[(str, Split)] = {}

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.SplitModel

    def __init__(self, *args, part_of, **kwargs):
        super().__init__(*args, part_of=part_of, **kwargs)

    def add_split(self, split_id=uuid.uuid4(), train_set=None, test_set=None, val_set=None):
        if val_set is None:
            val_set = []
        if test_set is None:
            test_set = []
        if train_set is None:
            train_set = []
        split = self.SplitModel.Split(train_set=train_set, test_set=test_set,
                                      validation_set=val_set)
        self.splits.update({str(split_id): split})


class SplitILF(MultiInjectionLogger):
    """
    Function logging the dataset splits
    """

    name = "SplitLogger"
    category = "SplitLogger"

    class SplitsILFOutput(OutputModel):
        category: str = "SplitILF-Output"
        splits: str = None

        class Config:
            orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.SplitsILFOutput

    @staticmethod
    def finalize_output(pads, *args, **kwargs):
        split_tracker = pads.cache.run_get(pads.cache.run_get("split_tracker"))
        call = split_tracker.get("call")
        output = split_tracker.get("output")
        to = split_tracker.get("TO")
        to.store(output, "splits")
        call.output = output.store()
        call.store()

    def __call_wrapped__(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _logger_call, _logger_output, _args,
                         _kwargs):
        """

        :param ctx:
        :param args:
        :param _pypads_result:
        :param kwargs:
        :return:
        """

        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        pads.cache.run_add("split_tracker", id(self))
        _return, time = OriginalExecutor(fn=_pypads_env.callback)(*_args, **_kwargs)

        if isinstance(_return, GeneratorType):
            items = list(_return)

            def generator():
                split_tracker = pads.cache.run_get(pads.cache.run_get("split_tracker"))
                call = split_tracker.get("call")
                output = split_tracker.get("output")
                pads.cache.add("tracking_mode","single")
                logger.info("Detected splitting, Tracking splits started...")
                if output.splits is None:
                    splits = SplitTO(part_of=output)
                else:
                    splits = output.splits
                for r in items:
                    split_id = uuid.uuid4()
                    pads.cache.run_add("current_split", split_id)
                    train, test, val, num = splitter_output(r, fn=_pypads_env.callback)
                    splits.add_split(split_id, train, test, val)
                    # splits.store(output, "splits")
                    pads.cache.run_add(pads.cache.run_get("split_tracker"),
                                       {'call': call, 'output': output, 'TO': splits})
                    yield r
        else:
            def generator():
                logger.info("Detected splitting, Tracking splits started...")
                split_tracker = pads.cache.run_get(pads.cache.run_get("split_tracker"))
                call = split_tracker.get("call")
                output = split_tracker.get("output")
                pads.cache.add("tracking_mode", "single")
                if output.splits is None:
                    splits = SplitTO(part_of=output)
                else:
                    splits = output.splits
                train, test, val, num = splitter_output(_return, fn=_pypads_env.callback)
                split_id = uuid.uuid4()
                pads.cache.run_add("current_split", split_id)
                splits.add_split(split_id, train, test, val)
                # splits.store(output, "splits")
                pads.cache.run_add(pads.cache.run_get("split_tracker"), {'call': call, 'output': output, 'TO': splits})
                return _return

        return generator(), time


class SplitILFTorch(MultiInjectionLogger):
    """
    Function logging splits used by torch DataLoader
    """
    name = "SplitTorchLogger"
    category = "Splitogger"

    supported_libraries = {LibSelector(name="torch", constraint="*", specificity=1)}

    def get_model_cls(cls) -> Type[BaseModel]:
        return SplitILF.SplitsILFOutput

    def _handle_error(self, *args, ctx, _pypads_env, error, **kwargs):
        if isinstance(error, StopIteration):
            logger.warning("Ignoring recovery of this StopIteration error: {}".format(error))
            original = _pypads_env.call.call_id.context.original(_pypads_env.callback)
            return original(ctx, *args, **kwargs)
        else:
            super()._handle_error(*args, ctx, _pypads_env, error, **kwargs)

    @staticmethod
    def finalize_output(pads, *args, **kwargs):
        split_tracker = pads.cache.run_get(pads.cache.run_get("split_tracker"))
        call = split_tracker.get("call")
        output = split_tracker.get("output")
        to = split_tracker.get("TO")
        to.store(output, "splits")
        call.output = output.store(split_tracker.get("base_path"))
        call.store()

    def __post__(self, ctx, *args, _logger_call, _pypads_pre_return, _pypads_result, _logger_output, _args, _kwargs,
                 **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        pads.cache.run_add("split_tracker", id(self))
        if _logger_output.splits is None:
            splits = SplitTO(part_of=_logger_output)
        else:
            splits = _logger_output.splits
        logger.info("Detected splitting, Tracking splits started...")
        train, test, val, num = splitter_output(_pypads_result, fn=ctx)
        split_id = uuid.uuid4()
        pads.cache.run_add("current_split", split_id)
        splits.add_split(split_id, train, test, val)
        # splits.store(_logger_output, "splits")
        pads.cache.run_add(pads.cache.run_get("split_tracker"),
                           {'call': _logger_call, 'output': _logger_output, 'TO': splits})
