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


def wikifier_enrichment(dataset, option, refined, stop_words):
    wikifier_url = os.getenv("WIKIFIER_BASE_API_URL")
    for index, row in dataset.iterrows():
        logging.info("Index = %s", index)
        word_tokens_title = word_tokenize(str(row["title"]))
        word_tokens_abstract = word_tokenize(str(row["abstract"]))
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
        title_data = {
            "text": row["title"],
            "lang": "en",
            "userKey": os.getenv("WIKIFIER_API_KEY"),
            "pageRankSqThreshold": 0.8,
        }

        abstract_data = {
            "text": row["abstract"],
            "lang": "en",
            "userKey": os.getenv("WIKIFIER_API_KEY"),
            "pageRankSqThreshold": 0.8,
        }

        title_response = requests.post(wikifier_url, data=title_data)
        abstract_response = requests.post(wikifier_url, data=abstract_data)

        title_linked_entities = [
            (ann["title"], ann["url"]) for ann in title_response.json()["annotations"]
        ]
        abstract_linked_entities = [
            (ann["title"], ann["url"])
            for ann in abstract_response.json()["annotations"]
        ]
        dataset.at[index, "wikidata_title_entities"] = list(title_linked_entities)
        dataset.at[index, "wikidata_abstract_entities"] = list(abstract_linked_entities)
        dataset.at[index, "wikidata_entities"] = list(
            set(list(abstract_linked_entities) + list(title_linked_entities))
        )
        dataset.at[index, "title"] = row["title"]
        dataset.at[index, "abstract"] = row["abstract"]
    return dataset


def tagme_enrichment(dataset, option, refined, stop_words):
    for index, row in dataset.iterrows():
        logging.info("Index = %s", index)
        word_tokens_title = word_tokenize(str(row["title"]))
        word_tokens_abstract = word_tokenize(str(row["abstract"]))
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
        title_params = {
            "text": row["title"],
            "lang": "en",
            "gcube-token": os.getenv("TAGME_API_KEY"),
        }

        abstract_params = {
            "text": row["abstract"],
            "lang": "en",
            "gcube-token": os.getenv("TAGME_API_KEY"),
        }

        title_response = requests.get(
            os.getenv("TAGME_BASE_API_URL"), params=title_params
        )
        abstract_response = requests.get(
            os.getenv("TAGME_BASE_API_URL"), params=abstract_params
        )
        title_annotations = title_response.json()
        abstract_annotations = abstract_response.json()

        title_ann = title_annotations["annotations"]
        abstract_ann = abstract_annotations["annotations"]
        abstract = row["abstract"]
        title = row["title"]

        logging.info(f"Title {title}")
        logging.info(f"Title annotations {title_ann}")
        logging.info(f"Abstract {abstract}")
        logging.info(f"Abstract annotations {abstract_ann}")

        title_linked_entities = [
            (ann["id"]) for ann in title_annotations["annotations"]
        ]
        abstract_linked_entities = [
            (ann["id"]) for ann in abstract_annotations["annotations"]
        ]
        dataset.at[index, "wikidata_title_entities"] = list(title_linked_entities)
        dataset.at[index, "wikidata_abstract_entities"] = list(abstract_linked_entities)
        dataset.at[index, "wikidata_entities"] = list(
            set(list(abstract_linked_entities) + list(title_linked_entities))
        )
        dataset.at[index, "title"] = row["title"]
        dataset.at[index, "abstract"] = row["abstract"]
    return dataset
