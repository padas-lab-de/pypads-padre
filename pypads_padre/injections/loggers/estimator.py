from typing import Type, Union, Optional

from pydantic import BaseModel
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import InjectionLogger
from pypads.app.injections.tracked_object import TrackedObject
from pypads.importext.versioning import all_libs
from pypads.model.logger_call import InjectionLoggerCallModel
from pypads.model.logger_output import TrackedObjectModel, OutputModel
from pypads.model.models import BaseStorageModel, ResultType
from pypads.utils.logging_util import data_str, data_path
from pypads.utils.util import persistent_hash


class EstimatorRepositoryObject(BaseStorageModel):
    """
    Class to be used in the repository holding an estimator. Repositories are supposed to store objects used over
    multiple runs.
    """
    name: str = ...  # Name for the estimator
    description: str = ...  # Description
    documentation: str = ...  # Extracted documentation
    parameter_schema: Union[str, dict] = ...  # Schema for parameters
    location: str = ...  # Place where it is defined
    category: str = "EstimatorRepositoryEntry"
    storage_type: Union[str, ResultType] = "estimator"


class EstimatorILFOutput(OutputModel):
    """
    Output of the logger. An output can reference multiple Tracked Objects or Values directly. In this case a own
    tracked object doesn't give a lot of benefit but enforcing a description a name and a category and could be omitted.
    """
    category: str = "EstimatorILF-Output"
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
        return EstimatorILFOutput

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

        # Get data from mapping file
        mapping_data = _pypads_env.data
        estimator_data = data_str(mapping_data, "estimator", "@schema", default={})

        # Create repository object
        ero = EstimatorRepositoryObject(
            name=data_str(estimator_data, "rdfs:label", default=ctx.__class__.__name__,
                          warning=f"No name given for {ctx.__class__}. "
                                  f"Extracting name from class."),
            description=data_str(estimator_data, "rdfs:description",
                                 default="Some unknown estimator."),
            documentation=data_str(estimator_data, "padre:documentation",
                                   default=ctx.__class__.__doc__,
                                   warning=f"No documentation defined on the mapping file for {ctx.__class__}. "
                                           f"Taking code documentation instead."),
            parameter_schema=data_path(estimator_data, "padre:parameters", default="unkown",
                                       warning=f"No parameters are defined on the mapping file for {ctx.__class__}. "
                                               f"Logging estimator without parameters."),
            location=_logger_call.original_call.call_id.context.reference,
            additional_data=estimator_data)

        # Compile identifying hash
        hash_id = persistent_hash(ero.json())

        # Add to repo if needed
        if not _pypads_env.pypads.estimator_repository.has_object(uid=hash_id):
            repo_obj = _pypads_env.pypads.estimator_repository.get_object(uid=hash_id)
            repo_obj.log_json(ero)

        # Create referencing object
        eto = EstimatorTO(repository_reference=hash_id, repository_type=_pypads_env.pypads.estimator_repository.name,
                          parent=_logger_output, additional_data=mapping_data)

        # Store object
        _logger_output.estimator = eto.store()
