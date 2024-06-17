import logging

from dotenv import load_dotenv
from nltk.tokenize import word_tokenize

from online_news_classification.functions import text_clean

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


def enrich(dataset, option, refined, stop_words):
    for index, row in dataset.iterrows():
        logging.info("Index = %s", index)
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
                    w for w in word_tokens_abstract if w.lower() not in stop_words
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
                    if not text_clean.truecase(w, only_proper_nouns=False) in stop_words
                ]
                row["title"] = " ".join(filtered_title)
                word_tokens_abstract = word_tokenize(str(row["abstract"]))
                filtered_abstract = [
                    w
                    for w in word_tokens_abstract
                    if not text_clean.truecase(w, only_proper_nouns=False) in stop_words
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
                    if not text_clean.truecase(w, only_proper_nouns=True) in stop_words
                ]
                row["title"] = " ".join(filtered_title)
                word_tokens_abstract = word_tokenize(str(row["abstract"]))
                filtered_abstract = [
                    w
                    for w in word_tokens_abstract
                    if not text_clean.truecase(w, only_proper_nouns=True) in stop_words
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
        dataset.at[index, "title_entities"] = list(title_entities)
        dataset.at[index, "abstract_entities"] = list(abstract_entities)
        dataset.at[index, "entities"] = list(
            set(list(abstract_entities) + list(title_entities))
        )
        dataset.at[index, "title"] = row["title"]
        dataset.at[index, "abstract"] = row["abstract"]
    return dataset
