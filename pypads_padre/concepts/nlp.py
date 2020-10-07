import re

from pypads.utils.util import is_package_available


def preprocess(corpus):
    corpus = re.sub(r'[\s]+', ' ', corpus)
    corpus = re.sub(r'[\t]+', '', corpus)
    corpus = re.sub(r'[\n]+', '', corpus)
    pat = re.compile(r'([a-zA-Z][^\[\]\+\<\>\-\.!?]*[\.!?])', re.M)
    corpus = " ".join(pat.findall(corpus))
    return corpus


def name_to_words(label):
    label = re.sub(r".*([a-z])([A-Z]).*", r"\g<1> \g<2>", label)
    label = label.replace("_", " ")
    return label.replace(".", " ")


def ner_tagging(corpus):
    if is_package_available("spacy"):
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(corpus)
        nouns = set()
        for chunk in doc.noun_chunks:
            if "=" not in chunk.text and "." not in chunk.text:
                nouns.add(chunk.text)

        ents = set()
        for ent in doc.ents:
            if "=" not in ent.text and "." not in ent.text and "`" not in ent.text and "/" not in ent.text:
                ents.add(ent.text)

        return str(nouns), str(ents)

    elif is_package_available("nltk"):
        # TODO use nltk to find named entities https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
        pass