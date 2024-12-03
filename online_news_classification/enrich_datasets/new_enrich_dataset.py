import os.path
import dask.dataframe as dd

from dotenv import load_dotenv
from refined.inference.processor import Refined
import logging
import time
import torch
import ast
import pandas as pd
from online_news_classification.functions import setup
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

def safe_process_text(tokenized_text, refined, device):
    """Safely process text, avoiding errors on invalid input."""
    if not isinstance(tokenized_text, str) or not len(tokenized_text):
        # If tokenized_text is not a valid string, return an empty list
        return []
    return refined.process_text(tokenized_text)


def process_span_wikidata(spans):
    """Extract wikidata_entity_id from spans"""
    return [span.predicted_entity.wikidata_entity_id for span in spans if span.predicted_entity is not None and span.predicted_entity.wikidata_entity_id is not None]

def process_span_wikipedia(spans):
    """Extract wikidata_entity_id from spans"""
    return [span.predicted_entity.wikipedia_entity_title for span in spans if span.predicted_entity is not None and span.predicted_entity.wikipedia_entity_title is not None]


def get_wikipedia_page_title(cur, entities):
    select_query = """select * from public.wiki_mapper wm  where wikidata_id = %s"""
    result_entities = []

    if entities and entities != '[]':
        for entity in ast.literal_eval(str(entities)):
            result = cur.execute(select_query, (str(entity), ))
            result = cur.fetchall()
            if result:
                result_entities.append(result[0][1])
            else:
                logging.info(f"{entity}")
    return result_entities

def enrich_dataset(args, refined, device):
    """Read the CSV file in chunks, process entities, and enrich with Wikipedia titles."""

    ddf = dd.read_csv(args.input_file, sep=";")

    # Partition the data into chunks (~1000 documents per partition)
    rows_per_partition = 500
    npartitions = ddf.shape[0].compute() // rows_per_partition + 1
    logging.info(npartitions)
    ddf = ddf.repartition(npartitions=npartitions)

    # Step 1: Persist the Dask DataFrame in memory
    ddf = ddf.persist()

    # Step 2: Convert Dask DataFrame partitions into delayed objects (this will retain the order)
    partitions = ddf.to_delayed()

    for i, partition in enumerate(partitions):
        start_time = time.time()

        # Compute the partition to a Pandas DataFrame
        df_partition = partition.compute()

        # Move tokenized texts to GPU for processing
        tokenized_titles = df_partition['tokenized_title'].fillna('')
        tokenized_abstracts = df_partition['abstract'].fillna('')

        # Process both titles and abstracts in batches on GPU
        spans_titles = [safe_process_text(title, refined, device) for title in tokenized_titles]
        spans_abstracts = [safe_process_text(abstract, refined, device) for abstract in tokenized_abstracts]

        # Extract entity IDs from spans
        df_partition['wikidata_title_entities'] = [process_span_wikidata(spans) for spans in spans_titles]
        df_partition['wikidata_abstract_entities'] = [process_span_wikidata(spans) for spans in spans_abstracts]
        df_partition['wikidata_entities'] = df_partition['wikidata_title_entities'] + df_partition['wikidata_abstract_entities']
        df_partition['wikidata_entities'] = df_partition['wikidata_entities'].apply(
            lambda x: list(set(x)) if isinstance(x, list) else x
        )

        # Fetch Wikipedia titles from database
        df_partition['wikipedia_title_entities_text'] = [process_span_wikipedia(spans) for spans in spans_titles]
        df_partition["wikipedia_abstract_entities_text"] = [process_span_wikipedia(spans) for spans in spans_abstracts]
        df_partition['wikipedia_entities_text'] = df_partition['wikipedia_title_entities_text'] + df_partition['wikipedia_abstract_entities_text']
        df_partition['wikipedia_entities_text'] = df_partition['wikipedia_entities_text'].apply(
            lambda x: list(set(x)) if isinstance(x, list) else x
        )

        # Combine text and enriched entities
        df_partition['enriched_text'] = df_partition['tokenized_text'] + ' - ' + df_partition['wikipedia_entities_text'].apply(lambda x: ', '.join(x))

        # Save the enriched chunk
        df_partition.to_csv(args.output_file, mode="a", sep=";", index=False,header=not os.path.exists(args.output_file))
        logging.info("--- %s seconds ---" % (time.time() - start_time))

def main():
    args = setup.get_arg_parser_enrich_1().parse_args()
    start_time = setup.initialize(
        "enrich_" + args.dataset
    )
    # Set device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refined = Refined.from_pretrained(
        model_name="wikipedia_model_with_numbers", entity_set="wikipedia", device=str(device)
    )
    logging.info("Started")
    enrich_dataset(args, refined, device)

    logging.info("--- %s seconds ---" % (time.time() - start_time))



if __name__ == "__main__":
    main()