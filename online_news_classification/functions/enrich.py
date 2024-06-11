import logging

from dotenv import load_dotenv
from nltk.tokenize import word_tokenize

from . import text

load_dotenv()


def enrich(dataset, option, refined, nlp, stop_words):
    for index, row in dataset.iterrows():
        logging.info("Index = %s", index)
        title_entities = []
        abstract_entities = []
        match option:
            case 1:
                # lower case
                word_tokens_title = word_tokenize(str(row["title"]))
                filtered_title = [
                    w for w in word_tokens_title if not w.lower() in stop_words
                ]
                row["title"] = " ".join(filtered_title)
                word_tokens_abstract = word_tokenize(str(row["abstract"]))
                filtered_abstract = [
                    w for w in word_tokens_abstract if not w.lower() in stop_words
                ]
                row["abstract"] = " ".join(filtered_abstract)
                spans_title = refined.process_text(row["title"])
                spans_abstract = refined.process_text(row["abstract"])
            case 2:
                # not only proper nouns
                word_tokens_title = word_tokenize(str(row["title"]))
                filtered_title = [
                    w
                    for w in word_tokens_title
                    if not text.truecase(w, only_proper_nouns=False) in stop_words
                ]
                row["title"] = " ".join(filtered_title)
                word_tokens_abstract = word_tokenize(str(row["abstract"]))
                filtered_abstract = [
                    w
                    for w in word_tokens_abstract
                    if not text.truecase(w, only_proper_nouns=False) in stop_words
                ]
                row["abstract"] = " ".join(filtered_abstract)
                spans_title = refined.process_text(row["title"])
                spans_abstract = refined.process_text(row["abstract"])
            case 3:
                # only proper nouns
                word_tokens_title = word_tokenize(str(row["title"]))
                filtered_title = [
                    w
                    for w in word_tokens_title
                    if not text.truecase(w, only_proper_nouns=True) in stop_words
                ]
                row["title"] = " ".join(filtered_title)
                word_tokens_abstract = word_tokenize(str(row["abstract"]))
                filtered_abstract = [
                    w
                    for w in word_tokens_abstract
                    if not text.truecase(w, only_proper_nouns=True) in stop_words
                ]
                row["abstract"] = " ".join(filtered_abstract)
                spans_title = refined.process_text(row["title"])
                spans_abstract = refined.process_text(row["abstract"])
            case _:
                # original
                row["title"] = word_tokenize(str(row["title"]))
                filtered_title = [w for w in word_tokens_title if w not in stop_words]
                row["title"] = " ".join(filtered_title)
                row["abstract"] = str(row["abstract"])
                filtered_abstract = [
                    w for w in word_tokens_abstract if w not in stop_words
                ]
                row["abstract"] = " ".join(filtered_abstract)
                spans_title = refined.process_text(row["title"])
                try:
                    spans_abstract = refined.process_text(row["abstract"])
                except IndexError:
                    print("Index error.")
                    next
        for span in spans_title:
            if span.predicted_entity is not None:
                if span.predicted_entity.wikidata_entity_id is not None:
                    title_entities.append(span.predicted_entity.wikidata_entity_id)
                    text = span.predicted_entity.wikipedia_entity_title
                    text.replace(",", "")
        for span in spans_abstract:
            if span.predicted_entity is not None:
                if span.predicted_entity.wikidata_entity_id is not None:
                    abstract_entities.append(span.predicted_entity.wikidata_entity_id)
                    text = span.predicted_entity.wikipedia_entity_title
                    text.replace(",", "")
        dataset.at[index, "title_entities"] = title_entities
        dataset.at[index, "title_entities"] = list(title_entities)
        dataset.at[index, "abstract_entities"] = abstract_entities
        dataset.at[index, "abstract_entities"] = list(abstract_entities)
        dataset.at[index, "entities"] = list(
            set(list(abstract_entities) + list(title_entities))
        )
        dataset.at[index, "title"] = row["title"]
        dataset.at[index, "abstract"] = row["abstract"]
    return dataset
