import uuid
from typing import Type, Any, List, Union

from pydantic import HttpUrl, BaseModel
from pypads import logger
from pypads.app.injections.base_logger import TrackedObject
from pypads.app.injections.injection import InjectionLogger
from pypads.importext.mappings import LibSelector
from pypads.model.logger_output import TrackedObjectModel, OutputModel

from pypads_padre.concepts.util import _tolist


class SingleInstanceTO(TrackedObject):
    """
        Tracking Object class for single instance results
        """

    class SingleInstancesModel(TrackedObjectModel):
        uri: HttpUrl = "https://www.padre-lab.eu/onto/SingleInstanceResult"

        class DecisionModel(BaseModel):
            instance: Union[str, int] = ...
            truth: Union[str, int] = None
            prediction: Union[str, int] = ...
            probabilities: List[Union[float]] = []

            class Config:
                orm_mode = True
                arbitrary_types_allowed = True

        split_id: uuid.UUID = ...
        decisions: List[DecisionModel] = []

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.SingleInstancesModel

    def __init__(self, *args, split_id, part_of, **kwargs):
        super().__init__(*args, split_id=split_id, part_of=part_of, **kwargs)

    def add_decision(self, instance, truth, prediction, probabilities):
        self.decisions.append(
            self.SingleInstancesModel.DecisionModel(instance=self.check_type(instance),
                                                    truth=self.check_type(truth), prediction=self.check_type(prediction),
                                                                          probabilities=self.check_type(probabilities)))

    def check_type(self,value):
        if "int" in str(type(value)):
            return int(value)
        elif "float" in str(type(value)):
            return float(value)
        elif "str" in str(type(value)):
            return str(value)
        elif "bool" in str(type(value)):
            return bool(value)
        elif "array" in str(type(value)):
            value_ = []
            for v in value:
                value_.append(self.check_type(v))
            return value_
        return value


class SingleInstanceILF(InjectionLogger):
    """
    Function logging individual decisions
    """
    name = "SingleInstanceILF"
    uri = "https://www.padre-lab.eu/single-instance-logger"

    class SingleInstanceOuptut(OutputModel):
        is_a: HttpUrl = "https://www.padre-lab.eu/onto/SingleInstanceILF-Output"

        individual_decisions: str = None

        class Config:
            orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.SingleInstanceOuptut

    def __post__(self, ctx, *args, _logger_call, _pypads_pre_return, _pypads_result, _logger_output, _args, _kwargs,
                 **kwargs):
        """
        :param ctx:
        :param args:
        :param _pypads_result:
        :param kwargs:
        :return:
        """
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        preds = _pypads_result
        if pads.cache.run_exists("predictions"):
            preds = pads.cache.run_pop("predictions")

        # check if there is info about decision scores
        probabilities = None
        if pads.cache.run_exists("probabilities"):
            probabilities = pads.cache.run_pop("probabilities")

        # check if there is info on truth values
        targets = None
        if pads.cache.run_exists("targets"):
            targets = pads.cache.run_get("targets")

        # check if there exists information about the current split
        current_split = None
        split_id = None
        if pads.cache.run_exists("current_split"):
            split_id = pads.cache.run_get("current_split")
            splitter = pads.cache.run_get(pads.cache.run_get("split_tracker"))

            current_split = splitter.get("TO").splits.get(str(split_id), None)

        # depending on available info log the predictions
        if current_split is None:
            logger.warning("No split information were found in the cache of the current run, "
                           "individual decision tracking might be missing Truth values, try to decorate you splitter!")
        else:
            decisions = SingleInstanceTO(split_id=split_id, part_of=_logger_output)
            if current_split.test_set is not None:
                try:
                    for i, instance in enumerate(current_split.test_set):
                        prediction = preds[i]
                        probability_scores = []
                        if probabilities is not None:
                            probability_scores = _tolist(probabilities[i])
                        truth = None
                        if targets is not None:
                            truth = targets[instance]
                        decisions.add_decision(instance=instance, truth=truth, prediction=prediction,
                                               probabilities=probability_scores)
                    decisions.store(_logger_output, "individual_decisions")
                except Exception as e:
                    logger.warning("Could not log single instance decisions due to this error '%s'" % str(e))


class Decisions_sklearn(SingleInstanceILF):
    """
    Function getting the prediction scores from sklearn estimators
    """

    supported_libraries = {LibSelector(name="sklearn", constraint="*", specificity=1)}

    # identity =

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity = SingleInstanceILF.__name__

    def __pre__(self, ctx, *args,
                _logger_call, _logger_output, _args, _kwargs, **kwargs):
        """

        :param ctx:
        :param args:
        :param kwargs:
        :return:
        """
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        # check if the estimator computes decision scores
        probabilities = None
        predict_proba = None
        if hasattr(ctx, "predict_proba"):
            # TODO find a cleaner way to invoke the original predict_proba in case it is wrapped
            predict_proba = ctx.predict_proba
            if _logger_call.original_call.call_id.context.has_original(predict_proba):
                predict_proba = _logger_call.original_call.call_id.context.original(predict_proba)
        elif hasattr(ctx, "_predict_proba"):
            predict_proba = ctx._predict_proba
            if _logger_call.original_call.call_id.context.has_original(predict_proba):
                _logger_call.original_call.call_id.context.original(predict_proba)
        if hasattr(predict_proba, "__wrapped__"):
            predict_proba = predict_proba.__wrapped__
        try:
            probabilities = predict_proba(*_args, **_kwargs)
        except Exception as e:
            if isinstance(e, TypeError):
                try:
                    predict_proba = predict_proba.__get__(ctx)
                    probabilities = predict_proba(*_args, **_kwargs)
                except Exception as ee:
                    logger.warning("Couldn't compute probabilities because %s" % str(ee))
            else:
                logger.warning("Couldn't compute probabilities because %s" % str(e))
        finally:
            pads.cache.run_add("probabilities", probabilities)


class Decisions_keras(SingleInstanceILF):
    """
    Function getting the prediction scores from keras models
    """

    supported_libraries = {LibSelector(name="keras", constraint="*", specificity=1)}

    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.identity = SingleInstanceILF.__name__

    def __pre__(self, ctx, *args,
                _logger_call, _logger_output, _args, _kwargs, **kwargs):
        """

        :param ctx:
        :param args:
        :param kwargs:
        :return:
        """
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        probabilities = None
        try:
            probabilities = ctx.predict(*_args, **_kwargs)
        except Exception as e:
            logger.warning("Couldn't compute probabilities because %s" % str(e))

        pads.cache.run_add("probabilities", probabilities)


class Decisions_torch(SingleInstanceILF):
    """
    Function getting the prediction scores from torch models
    """

    supported_libraries = {LibSelector(name="torch", constraint="*", specificity=1)}

    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.identity = SingleInstanceILF.__name__

    def __post__(self, ctx, *args, _logger_call, _pypads_pre_return, _pypads_result, _logger_output, _args, _kwargs,
                 **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        if hasattr(ctx, "training") and ctx.training:
            pass
        else:
            pads.cache.run_add("probabilities", _pypads_result.data.numpy())
            pads.cache.run_add("predictions", _pypads_result.argmax(dim=1).data.numpy())

            return super().__post__(ctx, *args, _logger_call=_logger_call, _pypads_pre_return=_pypads_pre_return,
                                    _pypads_result=_pypads_result, _logger_output=_logger_output, _args=_args,
                                    _kwargs=_kwargs, **kwargs)
