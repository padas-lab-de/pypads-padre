import os
from typing import List, Any, Type

from pydantic import HttpUrl, BaseModel
from pypads import logger
from pypads.app.backends.repository import Repository
from pypads.app.injections.base_logger import TrackedObject
from pypads.app.injections.injection import InjectionLogger
from pypads.model.models import TrackedObjectModel, OutputModel, ArtifactMetaModel
from pypads.utils.logging_util import WriteFormats

from pypads_padre.concepts.dataset import Crawler
from pypads_padre.concepts.util import persistent_hash, get_by_tag


class DatasetRepository(Repository):

    def __init__(self, *args, **kwargs):
        """
        Repository holding all the relevant schema information
        :param args:
        :param kwargs:
        """
        super().__init__(*args, name="pypads_datasets", **kwargs)


class DatasetTO(TrackedObject):
    """
    Tracking Object logging the used dataset in your run.
    """

    class DatasetModel(TrackedObjectModel):
        uri: HttpUrl = "https://www.padre-lab.eu/onto/Dataset"

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
        data: ArtifactMetaModel = ...

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.DatasetModel

    def __init__(self, *args, tracked_by, name, shape, **kwargs):
        super().__init__(*args, tracked_by=tracked_by, name=name, number_of_instances=shape[0],
                         number_of_features=shape[1], **kwargs)

    def store_data(self, obj: Any, metadata, format):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        # get the repo or create new where datasets are stored
        repo = DatasetRepository()

        # get the current active run
        current_run = pads.api.active_run()

        # add data set if it is not already existing with name and hash check
        try:
            _hash = persistent_hash(str(obj))
        except Exception:
            logger.warning("Could not compute the hash of the dataset object, falling back to dataset name hash...")
            _hash = persistent_hash(str(self.name))

        _stored = get_by_tag("pypads.dataset.hash", str(_hash), repo.id)
        if not _stored:
            with repo.context() as run:
                dataset_id = run.info.run_id
                path = os.path.join(run.info.artifact_uri, self.name)
                pads.cache.run_add("dataset_id", dataset_id, current_run.info.run_id)
                pads.api.set_tag("pypads.dataset", self.name)
                pads.api.set_tag("pypads.dataset.hash", _hash)
                meta = ArtifactMetaModel(path=path, description="Dataset binary", format=format)
                pads.api.log_mem_artifact(self.name, obj, write_format=format, meta=meta)
                pads.api.log_mem_artifact("metadata", metadata)

            pads.api.set_tag("pypads.datasetID", dataset_id)
        else:
            # look for the existing dataset and reference it to the active run
            if len(_stored) > 1:
                logger.warning("multiple existing datasets with the same hash!!!")
            else:
                run = _stored.pop()
                dataset_id = run.info.run_id
                path = os.path.join(run.info.artifact_uri, self.name)
                meta = ArtifactMetaModel(path=path, description="Dataset binary", format=format)
                pads.api.set_tag("pypads.datasetID", dataset_id)

        # Fill the tracked object for the current run
        features = metadata.get("features", None)
        if features is not None:
            for name, type, default_target, range in features:
                self.features.append(
                    self.DatasetModel.Feature(name=name, type=type, default_target=default_target, range=range))
        self.data = meta

    def _get_artifact_path(self, name="data"):
        return super()._get_artifact_path(name)


class DatasetILF(InjectionLogger):
    """
    Function logging the wrapped dataset loader
    """
    name = "DatasetLogger"
    uri = "https://www.padre-lab.eu/onto/dataset-logger"

    class DatasetILFOutput(OutputModel):
        is_a: HttpUrl = "https://www.padre-lab.eu/onto/DatasetILF-Output"

        dataset: DatasetTO.get_model_cls() = ...

        class Config:
            orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.DatasetILFOutput

    def __post__(self, ctx, *args, _pypads_write_format=WriteFormats.pickle, _logger_call, _pypads_pre_return,
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
        pads.cache.run_add("data", data)
        pads.cache.run_add("shape", metadata.get("shape"))
        pads.cache.run_add("targets", targets)

        # setting the dataset object name
        if hasattr(obj, "name"):
            ds_name = obj.name
        elif pads.cache.run_exists("dataset_name") and pads.cache.run_get("dataset_name") is not None:
            ds_name = pads.cache.run_get("dataset_name")
        else:
            ds_name = _logger_call.original_call.call_id.wrappee.__qualname__

        # Look for metadata information given by the user when using the decorators
        if pads.cache.run_exists("dataset_meta"):
            metadata = {**metadata, **pads.cache.run_get("dataset_meta")}

        dataset = DatasetTO(tracked_by=_logger_call, name=ds_name, shape=metadata.get("shape"))

        dataset.store_data(data, metadata, _pypads_write_format)
        dataset.store(_logger_output, "dataset")
