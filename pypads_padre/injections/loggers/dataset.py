import os
from typing import List, Any, Type, Union

from pydantic import BaseModel
from pypads import logger
from pypads.app.backends.repository import BaseRepositoryObjectModel
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.base_logger import TrackedObject
from pypads.app.injections.injection import InjectionLogger
from pypads.importext.versioning import all_libs
from pypads.model.logger_call import InjectionLoggerCallModel

from pypads.model.logger_output import TrackedObjectModel, OutputModel
from pypads.utils.logging_util import FileFormats
from pypads_onto.arguments import ontology_uri

from pypads_padre.concepts.dataset import Crawler
from pypads_padre.concepts.util import persistent_hash, validate_type


class DatasetRepositoryObject(BaseRepositoryObjectModel):
    """
    Class to be used in the repository holding a dataset. Repositories are supposed to store objects used over
    multiple runs.
    """
    name: str = ...  # Name of the dataset
    category: str = "DatasetRepository"
    description: str = ...
    documentation: str = ...
    binary_reference: str = ...  # Reference to the dataset binary
    location: str = ...  # Place where it is defined


class DatasetOutput(OutputModel):
    """
    Output of the logger
    """
    dataset: str = ...  # Reference to dataset TO


class DatasetTO(TrackedObject):
    """
    Tracking Object logging the used dataset in your run.
    """

    class DatasetModel(TrackedObjectModel):
        """
        Model defining the values for the tracked object.
        """

        context: Union[List[str], str] = str({
            "number_of_instances": {
                "@id": f"{ontology_uri}has_instances",
                "@type": f"{ontology_uri}DatasetProperty"
            },
            "number_of_features": {
                "@id": f"{ontology_uri}has_features",
                "@type": f"{ontology_uri}DatasetProperty"
            },
            "features": {
                "type": {
                    "@id": f"{ontology_uri}has_type",
                    "@type": f"{ontology_uri}FeatureProperty"
                },
                "default_target": {
                    "@id": f"{ontology_uri}is_target",
                    "@type": f"{ontology_uri}FeatureProperty"
                }
            },
            "data": {
                "@id": f"{ontology_uri}stored_at",
                "@type": f"{ontology_uri}Data"
            }
        })

        class Feature(BaseModel):
            name: str = ...
            type: str = ...
            default_target: bool = False
            range: tuple = None

            class Config:
                orm_mode = True

        category: str = "Dataset"
        name: str = ...
        description = "This tracked object references a dataset used in the experiment. "
        number_of_instances: int = ...
        number_of_features: int = ...
        features: List[Feature] = []
        repository_reference: str = ...  # reference to the dataset in the repository
        repository_type: str = ...  # type of the repository. Will always be extracted from the repository aka
        # 'pypads_datasets'

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.DatasetModel

    def __init__(self, *args, parent, name, shape, metadata, **kwargs):
        super().__init__(*args, parent=parent, name=name, number_of_instances=shape[0],
                         number_of_features=shape[1], **kwargs)
        features = metadata.get("features", None)
        if features is not None:
            for name, type, default_target, range in features:
                self.features.append(
                    self.DatasetModel.Feature(name=validate_type(name), type=validate_type(type),
                                              default_target=default_target,
                                              range=validate_type(range)))

    def store_data(self, obj: Any, metadata, format):
        # Fill the tracked object for the current run
        return self.store_mem_artifact(self.name, obj, write_format=format, description="Dataset binary",
                                       additional_data=metadata)


class DatasetILF(InjectionLogger):
    """
    Function logging the wrapped dataset loader.

        Hook:
        Hook this logger to the loader of a dataset (it can be a function, or class)
    """
    name = "Dataset Logger"
    category: str = "DatasetLogger"
    supported_libraries = {all_libs}

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return DatasetOutput

    def __post__(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _logger_call: InjectionLoggerCallModel,
                 _logger_output, _pypads_result, _args, _kwargs, _pypads_write_format=FileFormats.pickle, **kwargs):
        pads = _pypads_env.pypads

        # if the return object is None, take the object instance ctx
        dataset_object = _pypads_result if _pypads_result is not None else ctx

        # Get additional arguments if given by the user
        _dataset_kwargs = dict()
        if pads.cache.run_exists("dataset_kwargs"):
            _dataset_kwargs = pads.cache.run_get("dataset_kwargs")

        # Scrape the data object
        crawler = Crawler(dataset_object, ctx=_logger_call.original_call.call_id.context.container,
                          callback=_logger_call.original_call.call_id.wrappee,
                          kw=_kwargs)
        data, metadata, targets = crawler.crawl(**_dataset_kwargs)
        pads.cache.run_add("targets", targets)

        # Look for metadata information given by the user when using the decorators
        if pads.cache.run_exists("dataset_meta"):
            metadata = {**metadata, **pads.cache.run_get("dataset_meta")}

        # getting the dataset object name
        if hasattr(dataset_object, "name"):
            ds_name = dataset_object.name
        elif pads.cache.run_exists("dataset_name") and pads.cache.run_exists("dataset_name") is not None:
            ds_name = pads.cache.run_get("dataset_name")
        else:
            ds_name = _logger_call.original_call.call_id.wrappee.__qualname__

        # compile identifying hash
        try:
            data_hash = persistent_hash(str(dataset_object))
        except Exception:
            logger.warning("Could not compute the hash of the dataset object, falling back to dataset name hash...")
            data_hash = persistent_hash(str(self.name))

        # create referencing object
        dto = DatasetTO(parent=_logger_output, name=ds_name, shape=metadata.get("shape"), metadata=metadata,
                        repository_reference=data_hash, repository_type=_pypads_env.pypads.dataset_repository.name)

        # Add to repo if needed
        if not pads.dataset_repository.has_object(uid=data_hash):
            logger.info("Detected Dataset was not found in the store. Adding an entry...")
            repo_obj = pads.dataset_repository.get_object(uid=data_hash)
            binary_ref = repo_obj.log_mem_artifact(dto.name, dataset_object, write_format=_pypads_write_format,
                                                   description="Dataset binary",
                                                   additional_data=metadata, holder=dto)
            logger.info("Entry added in the dataset repository.")
            # create repository object
            dro = DatasetRepositoryObject(name=self.name,
                                          description="Some unkonwn Dataset",
                                          documentation=ctx.__doc__ if ctx else _logger_call.original_call.call_id.wrappee.__doc__,
                                          binary_reference=binary_ref,
                                          location=_logger_call.original_call.call_id.context.reference)
            repo_obj.log_json(dro)

        # Store object
        _logger_output.dataset = dto.store()
