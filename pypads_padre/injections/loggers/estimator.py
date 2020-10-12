from typing import Type, Union, Optional

from pydantic import BaseModel
from pypads import logger
from pypads.app.backends.repository import BaseRepositoryObjectModel
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import InjectionLogger
from pypads.app.injections.tracked_object import TrackedObject
from pypads.importext.versioning import all_libs
from pypads.model.logger_call import InjectionLoggerCallModel
from pypads.model.logger_output import TrackedObjectModel, OutputModel
from pypads.utils.util import persistent_hash


class EstimatorRepositoryObject(BaseRepositoryObjectModel):
    """
    Class to be used in the repository holding an estimator. Repositories are supposed to store objects used over
    multiple runs.
    """
    name: str = ...  # Name for the estimator
    description: str = ...  # Description
    documentation: str = ...  # Extracted documentation
    parameter_schema: Union[str, dict] = ...  # Schema for parameters
    location: str = ...  # Place where it is defined


class EstimatorOutput(OutputModel):
    """
    Output of the logger. An output can reference multiple Tracked Objects or Values directly. In this case a own
    tracked object doesn't give a lot of benefit but enforcing a description a name and a category and could be omitted.
    """
    estimator: str = ...  # Reference to estimator TO


class EstimatorTO(TrackedObject):
    """
    Tracking Object logging the used estimator in your run. In difference to outputs some values need to exist.
    """

    class EstimatorModel(TrackedObjectModel):
        """
        Model defining the values for the tracked object.
        """
        category = "InitializedEstimator"
        name = "Initialized Estimator"
        description = "This tracked object references a initialized estimator in the experiment. " \
                      "An estimator being initialized doesn't make sure it is also used."
        repository_reference: str = ...  # reference to the estimator in the repository
        repository_type: str = ...  # type of the repository. Will always be extracted from the repository aka
        # 'pypads_estimators'

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.EstimatorModel


class EstimatorILF(InjectionLogger):
    """
    Function tracking the instantiation of an estimator. This stores estimators as concepts into the repository and
    references them in the run. Mapping files should give data in the format of:

        Hook:
            Hook this logger the constructor of an estimator.
        Mapping_File:
            data:
                estimator:
                    name: "DecisionTree"
                    description: "This is a great tree."
                    other_names: []
                    parameters:
                        model_parameters:
                        - name: split_quality
                          kind_of_value: "{'gini', 'entropy'}"
                          optional: 'True'
                          description: The function to measure the quality of a split.
                          default_value: "'gini'"
                          path: criterion
                          ...
                      optimisation_parameters:
                        - name: presort
                          kind_of_value: "{boolean, 'auto'}"
                          optional: 'True'
                          description: Whether to presort the data to speed up the finding of best splits
                            in fitting.
                          default_value: "'auto'"
                          path: presort
                      execution_parameters: []
    """

    name = "Estimator Logger"
    category: str = "EstimatorLogger"
    supported_libraries = {all_libs}

    @classmethod
    def output_schema_class(cls) -> Optional[Type[OutputModel]]:
        return EstimatorOutput

    def __post__(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _logger_call: InjectionLoggerCallModel,
                 _logger_output, _pypads_result, **kwargs):
        """
        This function is used to extract estimator information from the code and the related mapping file.

        This is run after the hooked function is executed. Pypads injects a set of default parameters.
        :param ctx: A reference to the context on which the original function was called
        :param args: Args given to the original function
        :param _pypads_env: A logging environment object storing information about the used mappings, original_call etc.
        :param _logger_call: A information object storing additonal information about the logger call itself
        :param _logger_output: A prepared result object of the class defined in output_schema_class(cls)
        :param _pypads_result: The return value of the __pre__ function
        :param kwargs: Kwargs given to the original function
        :return:
        """

        mapping_data = _pypads_env.data

        # Warn about partial mappings
        if 'estimator' not in mapping_data or 'parameters' not in mapping_data['estimator']:
            logger.warning("No parameters are defined on the mapping file for " + str(
                ctx.__class__) + ". Logging estimator without parameters.")

        if 'estimator' not in mapping_data or 'name' not in mapping_data['estimator']:
            logger.warning("No name given for " + str(
                ctx.__class__) + ". Extracting name from class.")

        if 'estimator' not in mapping_data or 'description' not in mapping_data['estimator']:
            logger.warning("No description given for " + str(
                ctx.__class__) + ". Saving without name.")

        # Create repository object
        estimator_data = mapping_data['estimator'] if 'estimator' in mapping_data else {}
        ero = EstimatorRepositoryObject(name=estimator_data[
            'name'] if 'name' in estimator_data else ctx.__class__.__name__,
                                        description=estimator_data[
                                            'description'] if 'description' in estimator_data else
                                        "Some unknown estimator.",
                                        documentation=ctx.__class__.__doc__,
                                        parameter_schema=estimator_data[
                                            'parameters'] if 'parameters' in estimator_data else "unknown",
                                        location=_logger_call.original_call.call_id.context.reference)

        # Compile identifying hash
        hash_id = persistent_hash(ero.json())

        # Add to repo if needed
        if not _pypads_env.pypads.estimator_repository.has_object(uid=hash_id):
            repo_obj = _pypads_env.pypads.estimator_repository.get_object(uid=hash_id)
            repo_obj.log_json(ero)

        # Create referencing object
        eto = EstimatorTO(repository_reference=hash_id, repository_type=_pypads_env.pypads.estimator_repository.name,
                          parent=_logger_output)

        # Store object
        _logger_output.estimator = eto.store()
