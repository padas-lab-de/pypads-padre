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


class SplitTO(TrackedObject):
    """
    Tracking Object class for splits of your tracked dataset
    """

    class SplitModel(TrackedObjectModel):
        uri: HttpUrl = "https://www.padre-lab.eu/onto/Split"

        class Split(BaseModel):
            split_id: uuid.UUID = ...
            train_set: Union[List, Set, numpy.ndarray] = ...
            test_set: Union[List, Set, numpy.ndarray] = ...

            class Config:
                orm_mode = True
                arbitrary_types_allowed = True

        splits: List[Split] = []

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.SplitModel

    def __init__(self, *args, tracked_by, **kwargs):
        super().__init__(*args, tracked_by=tracked_by, **kwargs)

    def add_split(self, train_set=None, test_set=None):
        split = self.SplitModel.Split(split_id=uuid.uuid4(), train_set=train_set, test_set=test_set)
        self.splits.append(split)


class SplitILF(InjectionLogger):
    """
    Function logging the dataset splits
    """

    name = "SplitLogger"
    uri = "https://www.padre-lab.eu/onto/split-logger"

    class SplitsILFOutput(OutputModel):
        is_a: HttpUrl = "https://www.padre-lab.eu/onto/SplitILF-Output"
        splits: SplitTO.get_model_cls() = ...

        class Config:
            orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.SplitsILFOutput

    def __call_wrapped__(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _args, _kwargs):
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


class SplitILFTorch(InjectionLogger):
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


    def __post__(self, ctx, *args, _logger_call, _pypads_pre_return, _pypads_result, _logger_output, _args, _kwargs,
                 **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        if _logger_output.splits is None:
            split = SplitTO(tracked_by=_logger_call)

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
