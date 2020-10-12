import os
from typing import Type, Optional

from pydantic import BaseModel
from pypads.app.call import Call
from pypads.app.injections.base_logger import TrackedObject
from pypads.app.injections.injection import MultiInjectionLogger, MultiInjectionLoggerCall
from pypads.model.logger_output import OutputModel, TrackedObjectModel

from pypads_padre.concepts.nlp import preprocess, ner_tagging, name_to_words


class ExtractedDocs(TrackedObject):
    """
    Tracking object logging extracted Named entities from the documentation
    of used functions to reference concepts from the ontology
    """

    class DocModel(TrackedObjectModel):
        """
        Model defining the values of the ExtractedDocs tracked object.
        """
        category: str = "ExtractedConcepts"
        name: str = "ParsedDocs"
        description = "Nouns and Named Entities extracted " \
                      "using NER and POS tagging on the documentation of every tracked class/function."
        nouns: str = ...
        named_entities: str = ...

        class Config:
            orm_mode = True

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.DocModel

    def __init__(self, *args, parent, **kwargs):
        super().__init__(*args, parent=parent, **kwargs)
        self.doc_map = {}

    def add_docs(self, call: Call):
        if call.call_id.wrappee.__doc__:
            name = call.call_id.wrappee.__name__ + ".__doc__"
            self.doc_map[name] = call.call_id.wrappee.__doc__

        if call.call_id.context.container.__doc__:
            name = call.call_id.context.container.__name__ + ".__doc__"
            self.doc_map[name] = call.call_id.context.container.__doc__

        # Add ctx name to doc_map for named entity searching
        self.doc_map[call.call_id.context.container.__name__ + "_exists"] = "The " + name_to_words(
            call.call_id.context.container.__name__) + " exists."
        self.doc_map[call.call_id.wrappee.__name__ + "_exists"] = "The " + name_to_words(
            call.call_id.wrappee.__name__) + " exists."
        self.doc_map[call.call_id.wrappee.__name__ + "_is_in"] = "The " + name_to_words(
            call.call_id.wrappee.__name__) + " is in " + name_to_words(
            call.call_id.context.container.__name__) + "."


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

    @classmethod
    def output_schema_class(cls) -> Optional[Type[OutputModel]]:
        return cls.DocExtractionOutput

    @staticmethod
    def finalize_output(pads, logger_call, output, *args, **kwargs):
        to = output.docs
        docs = to.doc_map
        corpus = " ".join([doc for name, doc in docs.items()])
        corpus = preprocess(corpus)

        nouns, entities = ner_tagging(corpus)

        to.nouns = nouns
        to.entities = entities
        output.docs = to.store()
        logger_call.output = output.store()

    def __pre__(self, ctx, *args, _pypads_write_format=None, _logger_call: MultiInjectionLoggerCall, _logger_output,
                _args, _kwargs,
                **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        pads.cache.run_add("doc_parser", id(self))
        if _logger_output.docs is None:
            docs = ExtractedDocs(parent=_logger_output)
        else:
            docs = _logger_output.docs
        docs.add_docs(_logger_call.last_call)
        # !Add ctx name to doc_map for named entity searching
        _logger_output.docs = docs
