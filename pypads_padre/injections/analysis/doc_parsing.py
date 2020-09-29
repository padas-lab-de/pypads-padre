import os

from pydantic import BaseModel
from pypads.app.injections.base_logger import LoggerCall, TrackedObject
from pypads.app.injections.injection import MultiInjectionLogger
from pypads.model.logger_output import OutputModel, TrackedObjectModel
from typing import Type

from pypads_padre.concepts.nlp import preprocess, ner_tagging, name_to_words


class ExtractedDocs(TrackedObject):
    """
    Tracking object logging extracted Named entities from the documentation of used functions to reference concepts from the ontology
    """

    class DocModel(TrackedObjectModel):
        category: str = "ExtractedConcepts"

        nouns: str = ...
        named_entities: str = ...

        class Config:
            orm_mode = True

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.DocModel

    def __init__(self, *args, part_of, **kwargs):
        super().__init__(*args, part_of=part_of, **kwargs)
        self.doc_map = {}

    def add_docs(self, call: LoggerCall):
        if call.original_call.call_id.wrappee.__doc__:
            name = os.path.join(call.original_call.to_folder(),
                                call.original_call.call_id.wrappee.__name__ + ".__doc__")
            self.doc_map[name] = call.original_call.call_id.wrappee.__doc__

        if call.original_call.call_id.context.container.__doc__:
            name = os.path.join(call.original_call.to_folder(),
                                call.original_call.call_id.context.container.__name__ + ".__doc__")
            self.doc_map[name] = call.original_call.call_id.context.container.__doc__

        # Add ctx name to doc_map for named entity searching
        self.doc_map[call.original_call.call_id.context.container.__name__ + "_exists"] = "The " + name_to_words(
            call.original_call.call_id.context.container.__name__) + " exists."
        self.doc_map[call.original_call.call_id.wrappee.__name__ + "_exists"] = "The " + name_to_words(
            call.original_call.call_id.wrappee.__name__) + " exists."
        self.doc_map[call.original_call.call_id.wrappee.__name__ + "_is_in"] = "The " + name_to_words(
            call.original_call.call_id.wrappee.__name__) + " is in " + name_to_words(
            call.original_call.call_id.context.container.__name__) + "."


class DocExtractionILF(MultiInjectionLogger):
    """
    Function logging extracted concepts from documentation.
    """

    name = "DocsExtractor"
    category: str = "DocExtractionLogger"

    class DocExtractionOutput(OutputModel):
        category: str = "DocExtractionILF-Output"

        docs: str = None

        class Config:
            orm_mode = True

    @staticmethod
    def finalize_output(pads, *args, **kwargs):

        doc_tracker = pads.cache.run_get(pads.cache.run_get("doc_tracker"))
        call = doc_tracker.get("call")
        output = doc_tracker.get("output")
        to = output.docs
        docs = to.doc_map
        corpus = " ".join([doc for name, doc in docs.items()])
        corpus = preprocess(corpus)

        nouns, entities = ner_tagging(corpus)

        to.nouns = nouns
        to.entities = entities
        to.store(output, "docs")
        call.output = output.store()
        call.store()

    def __pre__(self, ctx, *args, _pypads_write_format=None, _logger_call: LoggerCall, _logger_output, _args, _kwargs,
                **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        pads.cache.run_add("doc_parser", id(self))
        if _logger_output.docs is None:
            docs = ExtractedDocs(part_of=_logger_output)
        else:
            docs = _logger_output.docs
        docs.add_docs(_logger_call)
        # !Add ctx name to doc_map for named entity searching
        _logger_output.docs = docs
