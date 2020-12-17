import uuid
from typing import Type, List, Union

from pydantic import BaseModel, Field
from pypads import logger
from pypads.app.injections.base_logger import TrackedObject
from pypads.app.injections.injection import InjectionLogger
from pypads.importext.versioning import LibSelector
from pypads.model.logger_output import TrackedObjectModel, OutputModel
from pypads.model.models import BaseStorageModel, ResultType, IdReference
# from pypads_onto.arguments import ontology_uri

from pypads_padre.concepts.util import _tolist, validate_type, _len

ontology_uri = "https://www.padre-lab.eu/onto/"


class SingleInstanceTO(TrackedObject):
    """
        Tracking Object class logging instance based results/decisions of your model.
    """

    class SingleInstancesModel(TrackedObjectModel):
        """
        Model defining the values for the tracked object
        """
        context: Union[List[str], str, dict] = {
            "split_id": {
                "@id": f"{ontology_uri}of_split",
                "@type": "rdf:xsd:string"
            }
        }

        class DecisionModel(BaseStorageModel):
            """
            Model defining the values for a individual model decision.
            """
            context: Union[List[str], str, dict] = Field(alias="@context", default={
                "instance": {
                    "@id": f"{ontology_uri}is_instance",
                    "@type": "rdf:XMLLiteral"
                },
                "truth": {
                    "@id": f"{ontology_uri}labeled_as",
                    "@type": "rdf:XMLLiteral"
                },
                "prediction": {
                    "@id": f"{ontology_uri}predicted_as",
                    "@type": "rdf:XMLLiteral"
                }

            })
            category = "SingleDecision"
            instance: Union[str, int] = ...
            truth: Union[str, int] = None
            prediction: Union[str, int] = ...
            probabilities: List[float] = []
            storage_type: Union[str, ResultType] = "decisions"

            class Config:
                orm_mode = True
                arbitrary_types_allowed = True

        category: str = "InstanceBasedResults"
        name: str = "Instance Based Results"
        description = "Individual results of the model for each data sample, " \
                      "e.g, [{'instance': 1, 'truth': 2, 'predicted': 1, probabilities: [0.1,0.5,0.4]}]"
        split_id: str = ...
        decisions: List[DecisionModel] = []

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.SingleInstancesModel

    def __init__(self, *args, split_id: uuid.UUID, parent, **kwargs):
        super().__init__(*args, split_id=str(split_id), parent=parent, **kwargs)

    def add_decision(self, instance, truth, prediction, probabilities):
        self.decisions.append(
            self.SingleInstancesModel.DecisionModel(instance=validate_type(instance),
                                                    truth=validate_type(truth), prediction=validate_type(prediction),
                                                    probabilities=validate_type(probabilities)))


class SingleInstanceOuptut(OutputModel):
    """
    Output model of the SingleInstance Injection Logger.
    """
    individual_decisions: Union[List[IdReference], IdReference] = None

    class Config:
        orm_mode = True


class SingleInstanceILF(InjectionLogger):
    """
    Function logging individual decisions

        Hook:
            Hook this logger to the inference function of your model (predict, forward,...)
    """
    name = "SingleInstance"
    category = "SingleInstanceLogger"

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return SingleInstanceOuptut

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
        mode = None
        splits = None
        if pads.cache.run_exists("current_split"):
            split_id = pads.cache.run_get("current_split")
            splitter = pads.cache.run_get(pads.cache.run_get("split_tracker"))
            splits = splitter.get("output").splits.splits
            mode = pads.cache.get("tracking_mode", "single")
            current_split = splits.get(str(split_id), None)

        # depending on available info log the predictions
        if current_split is None:
            logger.warning("No split information were found in the cache of the current run, "
                           "individual decision tracking might be missing Truth values, try to decorate you splitter!")
        else:
            logger.info(
                "Logging single instance / individual decisions depending on the availability of split information, "
                "predictions, probabilites and target values.")
            if mode == "multiple" and _len(preds) == _len(targets):
                _logger_output.individual_decisions = []
                for split_id, split in splits.items():
                    decisions = SingleInstanceTO(split_id=uuid.UUID(split_id), parent=_logger_output)
                    if split.test_set is not None:
                        try:
                            for i, instance in enumerate(split.test_set):
                                prediction = preds[i]
                                probability_scores = []
                                if probabilities is not None:
                                    probability_scores = _tolist(probabilities[i])
                                truth = None
                                if targets is not None:
                                    truth = targets[instance]
                                decisions.add_decision(instance=instance, truth=truth, prediction=prediction,
                                                       probabilities=probability_scores)
                            _logger_output.individual_decisions.append(decisions.store())
                        except Exception as e:
                            logger.warning(
                                "Could not log single instance decisions due to this error '%s'" % str(e))
            else:
                decisions = SingleInstanceTO(split_id=split_id, parent=_logger_output)
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
                        _logger_output.individual_decisions = decisions.store()
                    except Exception as e:
                        logger.warning("Could not log single instance decisions due to this error '%s'" % str(e))


class DecisionsSklearnILF(SingleInstanceILF):
    """
    Function getting the prediction scores from sklearn estimators
        Hook:
            Hook this logger to the inference function of your model, i.e. sklearn.BaseEstimator.predict.
    """
    name = "Sklearn Decisions Logger"
    type = "SklearnDecisionsLogger"

    supported_libraries = {LibSelector(name="sklearn", constraint="*", specificity=1)}

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


class DecisionsKerasILF(SingleInstanceILF):
    """
    Function getting the prediction scores from keras models.
        Hook:
            Hook this logger to the inference function of your model, i.e. keras.engine.training.Model.predict_classes.
    """
    name = "Keras Decisions Logger"
    category = "KerasDecisionsLogger"

    supported_libraries = {LibSelector(name="keras", constraint="*", specificity=1)}

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

        probabilities = None
        try:
            probabilities = ctx.predict(*_args, **_kwargs)
        except Exception as e:
            logger.warning("Couldn't compute probabilities because %s" % str(e))

        pads.cache.run_add("probabilities", probabilities)


class DecisionsTorchILF(SingleInstanceILF):
    """
    Function getting the prediction scores from torch models.
        Hook:
            Hook this logger to the inference function of your model, e.g, torch.modules.container.Sequential.forward.
    """
    name = "PyTorch Decisions Logger"
    category = "TorchDecisionsLogger"

    supported_libraries = {LibSelector(name="torch", constraint="*", specificity=1)}

    def __init__(self, *args, **kwargs):
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
