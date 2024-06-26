import argparse
import logging
import os
import time

import spacy
from dotenv import load_dotenv
from nltk.corpus import stopwords
from refined.inference.processor import Refined

from online_news_classification import constants
from online_news_classification.functions import logs_config

load_dotenv()


def initialize(log_name):
    start_time = time.time()
    logs_config.create_log_file(log_name)
    return start_time


def finalize(start_time):
    logging.info("--- %s seconds ---" % (time.time() - start_time))
    logging.info("****** Finished ******")


def initilize_with_models(log_name):
    start_time = initialize(log_name)
    refined = Refined.from_pretrained(
        model_name="wikipedia_model_with_numbers", entity_set="wikipedia"
    )
    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words("english"))
    nlp.add_pipe("entityLinker", last=True)
    return start_time, refined, nlp, stop_words


def get_arg_parser_get_dataset_from_api():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, help=constants.START_DATE_HELP)
    parser.add_argument("--end_date", type=str, help=constants.END_DATE_HELP)
    parser.add_argument("--month", type=int, help="month")
    parser.add_argument("--year", type=int, help="year")
    return parser


def get_arg_parser_to_csv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help=constants.INPUT_PATH_HELP
    )
    parser.add_argument(
        "--output", type=str, required=True, help=constants.OUTPUT_PATH_HELP
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help="type of dataset (i): file, (ii): api",
    )
    parser.add_argument(
        "--convert_mode",
        type=str,
        default="folder",
        help="process all folder or simply a document",
    )
    return parser


def get_arg_parser_split():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help=constants.INPUT_FILE_SPLIT_HELP)
    parser.add_argument("--input_dir", type=str, help=constants.INPUT_FILE_SPLIT_HELP)
    parser.add_argument(
        "--out_dir", type=str, required=True, help="directory to put files"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help=constants.DATASET_SPLIT_HELP
    )
    parser.add_argument(
        "--num_lines", type=int, required=True, help="number of lines in each file"
    )
    return parser


def get_arg_parser_merge():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help=constants.INPUT_FILE_SPLIT_HELP
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="directory to put files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=constants.DATASET_MERGE_HELP,
    )
    return parser


def get_arg_parser_enrich():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="directory that have splitted files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="path to enriched file"
    )
    parser.add_argument(
        "--capitalization", type=int, default=0, help=constants.CAPITALIZATION_HELP
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help=constants.DATASET_ENRICH_HELP
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        required=True,
        help="name of the dataset to enrich",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=5,
        help="number of processes to make the parallel enrichment",
    )
    parser.add_argument(
        "--tmp_dir", type=str, required=True, help="path to enriched file"
    )
    parser.add_argument(
        "--enrich_mode",
        type=str,
        default="folder",
        help="process all folder or simply a document",
    )
    return parser


def get_arg_parser_classification():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="path to file to classify"
    )
    parser.add_argument("--tmp_dir", type=str, required=True, help="path to tmp folder")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="path to results folder"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help=constants.DATASET_CLASSIFICATION_HELP
    )
    parser.add_argument(
        "--capitalization", type=int, default=0, help=constants.CAPITALIZATION_HELP
    )
    parser.add_argument(
        "--feature_extraction",
        type=str,
        default="tf-idf",
        help="feature extraction option",
    )
    parser.add_argument(
        "--classification_type",
        type=str,
        default="hdt_non_adaptive",
        help="capitalization option",
    )
    parser.add_argument("--text", type=str, required=True, help="text option")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="original",
        help="type of dataset to use: (i) original or (ii) both",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="file",
        help="format of dataset to use: (i) file or (ii) API",
    )
    parser.add_argument(
        "--with_copy",
        type=str,
        default="no",
        help="copy the existing files to tmp folder",
    )
    return parser


def get_env_variable(var_name, default_value, cast_type):
    value = os.getenv(var_name, default_value)
    try:
        return cast_type(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {var_name} must be of type {cast_type.__name__}"
        )
