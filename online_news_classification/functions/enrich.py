import logging
import os

import requests
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize

from online_news_classification.functions import text

load_dotenv()


def process_span_title(spans_title):
    title_entities = []
    for span in spans_title:
        if span.predicted_entity is not None:
            if span.predicted_entity.wikidata_entity_id is not None:
                title_entities.append(span.predicted_entity.wikidata_entity_id)
    return title_entities


def process_span_abstract(spans_abstract):
    abstract_entities = []
    for span in spans_abstract:
        if span.predicted_entity is not None:
            if span.predicted_entity.wikidata_entity_id is not None:
                abstract_entities.append(span.predicted_entity.wikidata_entity_id)
    return abstract_entities


def refined_enrichment(dataset, option, refined, stop_words):
    for index, row in dataset.iterrows():
        logging.info("Index = %s", index)
        abstract_entities = []
        word_tokens_title = word_tokenize(str(row["title"]))
        word_tokens_abstract = word_tokenize(str(row["abstract"]))
        title_entities = []
        abstract_entities = []
        match option:
            case "lower":
                # lower case
                filtered_title = [
                    w for w in word_tokens_title if w.lower() not in stop_words
                ]
                row["title"] = " ".join(filtered_title)
                filtered_abstract = [
                    w for w in word_tokens_abstract if w.lower() not in stop_words
                ]
                row["abstract"] = " ".join(filtered_abstract)
                spans_title = refined.process_text(row["title"])
                spans_abstract = refined.process_text(row["abstract"])
            case "not_only_proper_nouns":
                # not only proper nouns
                filtered_title = [
                    w
                    for w in word_tokens_title
                    if text.truecase(w, only_proper_nouns=False) not in stop_words
                ]
                row["title"] = " ".join(filtered_title)
                filtered_abstract = [
                    w
                    for w in word_tokens_abstract
                    if text.truecase(w, only_proper_nouns=False) not in stop_words
                ]
                row["abstract"] = " ".join(filtered_abstract)
                spans_title = refined.process_text(row["title"])
                spans_abstract = refined.process_text(row["abstract"])
            case "only_proper_nouns":
                # only proper nouns
                filtered_title = [
                    w
                    for w in word_tokens_title
                    if text.truecase(w, only_proper_nouns=True) not in stop_words
                ]
                row["title"] = " ".join(filtered_title)
                filtered_abstract = [
                    w
                    for w in word_tokens_abstract
                    if text.truecase(w, only_proper_nouns=True) not in stop_words
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
        title_entities = process_span_title(spans_title)
        abstract_entities = process_span_abstract(spans_abstract)
        row["wikidata_title_entities"] = list(title_entities)
        row["wikidata_abstract_entities"] = list(abstract_entities)
        dataset.at[index, "wikidata_title_entities"] = list(title_entities)
        dataset.at[index, "wikidata_abstract_entities"] = list(abstract_entities)
        dataset.at[index, "wikidata_entities"] = list(
            set(list(abstract_entities) + list(title_entities))
        )
        dataset.at[index, "title"] = row["title"]
        dataset.at[index, "abstract"] = row["abstract"]
    return dataset