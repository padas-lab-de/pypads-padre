import uuid
from types import GeneratorType
from typing import Tuple, Iterable, List, Union, Set, Type

import numpy
from pydantic import HttpUrl, BaseModel
from pypads import logger
from pypads.app.injections.base_logger import TrackedObject
from pypads.app.injections.injection import InjectionLogger, OriginalExecutor, MultiInjectionLogger
from pypads.importext.mappings import LibSelector
from pypads.app.env import InjectionLoggerEnv
from pypads.injections.analysis.time_keeper import add_run_time, TimingDefined
from pypads.model.models import OutputModel, TrackedObjectModel


def splitter_output(result, fn):
    # TODO rework this function to return train, test, val indices

    # check if the output of the splitter is a tuple of indices
    if isinstance(result, Tuple):
        if "sklearn" in fn.__module__:
            return result[0].tolist(), result[1].tolist(), None, None
        elif "default_splitter" in fn.__name__:
            return result[1], result[2], result[3], result[0]
        else:
            if len(result) < 4:
                return result[0], result[1], result[3], None
    else:
        if "torch" in fn.__module__:
            return result, None, None, None
        else:
            return None


class SplitTO(TrackedObject):
    """
    Tracking Object class for splits of your tracked dataset
    """

    class SplitModel(TrackedObjectModel):
        uri: HttpUrl = "https://www.padre-lab.eu/onto/Split"

        class Split(BaseModel):
            split_id: uuid.UUID = ...
            train_set: List = []
            test_set: List = []
            validation_set: List = []

            class Config:
                orm_mode = True
                arbitrary_types_allowed = True

        splits: List[Split] = []

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.SplitModel

    def __init__(self, *args, tracked_by, **kwargs):
        super().__init__(*args, tracked_by=tracked_by, **kwargs)

    def add_split(self, split_id=uuid.uuid4(), train_set=None, test_set=None, val_set=None):
        if val_set is None:
            val_set = []
        if test_set is None:
            test_set = []
        if train_set is None:
            train_set = []
        split = self.SplitModel.Split(split_id=split_id, train_set=train_set, test_set=test_set,
                                      validation_set=val_set)
        self.splits.append(split)


class SplitILF(MultiInjectionLogger):
    """
    Function logging the dataset splits
    """

    name = "SplitLogger"
    uri = "https://www.padre-lab.eu/onto/split-logger"

    class SplitsILFOutput(OutputModel):
        is_a: HttpUrl = "https://www.padre-lab.eu/onto/SplitILF-Output"
        splits: SplitTO.get_model_cls() = None

        class Config:
            orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.SplitsILFOutput

    @staticmethod
    def store(pads, *args, **kwargs):
        split_tracker = pads.cache.run_get(pads.cache.run_get("split_tracker"))
        call = split_tracker.get("call")
        output = split_tracker.get("output")

        call.output = output.store(split_tracker.get("base_path"))
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
                if output.splits is None:
                    splits = SplitTO(tracked_by=call)
                else:
                    splits = output.splits
                for r in items:
                    split_id = uuid.uuid4()
                    pads.cache.run_add("current_split", split_id)
                    train, test, val, num = splitter_output(r, fn=_pypads_env.callback)
                    splits.add_split(split_id,train, test, val)
                    splits.store(output, "splits")
                    pads.cache.run_add(pads.cache.run_get("split_tracker"), {'call': call, 'output': output})
                    yield r
        else:
            def generator():
                split_tracker = pads.cache.run_get(pads.cache.run_get("split_tracker"))
                call = split_tracker.get("call")
                output = split_tracker.get("output")
                if output.splits is None:
                    splits = SplitTO(tracked_by=call)
                else:
                    splits = output.splits
                train, test, val, num = splitter_output(_return, fn=_pypads_env.callback)
                split_id = uuid.uuid4()
                pads.cache.run_add("current_split", split_id)
                splits.add_split(split_id, train, test, val)
                splits.store(output, "splits")
                pads.cache.run_add(pads.cache.run_get("split_tracker"), {'call': call, 'output': output})
                return _return

        return generator(), time


class SplitILFTorch(MultiInjectionLogger):
    """
    Function logging splits used by torch DataLoader
    """
    name = "SplitTorchLogger"
    uri = "https://www.padre-lab.eu/onto/split-torch-logger"

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
    def store(pads, *args, **kwargs):
        pass

    def __post__(self, ctx, *args, _logger_call, _pypads_pre_return, _pypads_result, _logger_output, _args, _kwargs,
                 **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        if _logger_output.splits is None:
            split = SplitTO(tracked_by=_logger_call)
        else:
            splits = _logger_output.splits

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
