import os
from typing import List, Any, Type, Union

from pydantic import HttpUrl, BaseModel
from pypads import logger
from pypads.app.injections.base_logger import TrackedObject
from pypads.app.injections.injection import InjectionLogger
from pypads_padre.arguments import ontology_uri
from pypads.model.logger_output import TrackedObjectModel, OutputModel
from pypads.utils.logging_util import FileFormats

from pypads_padre.app.backends.repository import DatasetRepository
from pypads_padre.concepts.dataset import Crawler
from pypads_padre.concepts.util import persistent_hash, get_by_tag, validate_type


class DatasetTO(TrackedObject):
    """
    Tracking Object logging the used dataset in your run.
    """

    class DatasetModel(TrackedObjectModel):
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

        category: str = "Dataset"

        class Feature(BaseModel):
            name: str = ...
            type: str = ...
            default_target: bool = False
            range: tuple = None

            class Config:
                orm_mode = True

        name: str = ...
        number_of_instances: int = ...
        number_of_features: int = ...
        features: List[Feature] = []
        data: str = ...

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.DatasetModel

    def __init__(self, *args, part_of, name, shape, **kwargs):
        super().__init__(*args, part_of=part_of, name=name, number_of_instances=shape[0],
                         number_of_features=shape[1], **kwargs)

    def store_data(self, obj: Any, metadata, format):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        # get the repo or create new where datasets are stored
        dataset_repo = DatasetRepository()

        # get the current active run
        current_run = pads.api.active_run()

        # add data set if it is not already existing with name and hash check
        try:
            data_hash = persistent_hash(str(obj))
        except Exception:
            logger.warning("Could not compute the hash of the dataset object, falling back to dataset name hash...")
            data_hash = persistent_hash(str(self.name))

        # _stored = get_by_tag("pypads.dataset.hash", str(_hash), dataset_repo.id)
        if not dataset_repo.has_object(uid=data_hash):
            logger.info("Detected Dataset was not found in the store. Adding an entry...")
            dataset_entity = dataset_repo.get_object(uid=data_hash)
            dataset_id = dataset_entity.run_id
            pads.cache.run_add("dataset_id", dataset_id, current_run.info.run_id)
            with dataset_repo.context(run_id=dataset_id) as run:
                self.store_tag("pypads.dataset", self.name, description="Dataset Name")
                self.store_tag("pypads.dataset.hash", data_hash, description="Dataset hash")
                self.data = self.store_artifact(self.name, obj, write_format=format, description="Dataset binary",
                                                meta=metadata)
                logger.info("Entry added in the dataset repository.")

            self.store_tag("pypads.datasetID", dataset_id, description="Dataset repository ID")
        else:
            logger.info("Detected dataset already exists in the store. Getting information about the entry...")
            dataset_entity = dataset_repo.get_object(uid=data_hash)
            # look for the existing dataset and reference it to the active run
            dataset_id = dataset_entity.run_id
            path = os.path.join(dataset_entity.run.info.artifact_uri, self.name)
            self.data = path
            self.store_tag("pypads.datasetID", dataset_id, description="Dataset repository ID")

        # Fill the tracked object for the current run
        features = metadata.get("features", None)
        if features is not None:
            for name, type, default_target, range in features:
                self.features.append(
                    self.DatasetModel.Feature(name=validate_type(name), type=validate_type(type), default_target=default_target,
                                              range=validate_type(range)))


class DatasetILF(InjectionLogger):
    """
    Function logging the wrapped dataset loader
    """
    name = "DatasetLogger"
    category = "DatasetLogger"

    class DatasetILFOutput(OutputModel):
        category: str = "DatasetILF-Output"

        dataset: str = None

        class Config:
            orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.DatasetILFOutput

    def __post__(self, ctx, *args, _pypads_write_format=FileFormats.pickle, _logger_call, _pypads_pre_return,
                 _pypads_result, _logger_output, _args, _kwargs,
                 **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        # if the return object is None, take the object instance ctx
        obj = _pypads_result if _pypads_result is not None else ctx

        # Get additional arguments if given by the user
        _dataset_kwargs = dict()
        if pads.cache.run_exists("dataset_kwargs"):
            _dataset_kwargs = pads.cache.run_get("dataset_kwargs")

        # Scrape the data object
        crawler = Crawler(obj, ctx=_logger_call.original_call.call_id.context.container,
                          callback=_logger_call.original_call.call_id.wrappee,
                          kw=_kwargs)
        data, metadata, targets = crawler.crawl(**_dataset_kwargs)
        pads.cache.run_add("targets", targets)

        # setting the dataset object name
        if hasattr(obj, "name"):
            ds_name = obj.name
        elif pads.cache.run_exists("dataset_name") and pads.cache.run_exists("dataset_name") is not None:
            ds_name = pads.cache.run_get("dataset_name")
        else:
            ds_name = _logger_call.original_call.call_id.wrappee.__qualname__

        # Look for metadata information given by the user when using the decorators
        if pads.cache.run_exists("dataset_meta"):
            metadata = {**metadata, **pads.cache.run_get("dataset_meta")}

        dataset = DatasetTO(part_of=_logger_output, name=ds_name, shape=metadata.get("shape"))

        dataset.store_data(data, metadata, _pypads_write_format)
        dataset.store(_logger_output, "dataset")
