import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from online_news_classification import constants

load_dotenv()


def generate_plot_image(
    docs_number,
    preq,
    preq_a,
    preq_w,
    drifts,
    dataset_name,
    dataset_type,
    file_name,
    classifier_type,
    feature_type,
    enrichment_type,
):
    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(docs_number), preq, label="Prequential")
    ax.plot(range(docs_number), preq_a, label="Prequential Alpha")
    ax.plot(range(docs_number), preq_w, label="Prequential Window")
    for d in drifts:
        plt.axvline(x=d["index"], color="r")
    ax.legend()
    plots_folder = os.getenv("PLOTS_FOLDER")
    plt.savefig(
        f"{plots_folder}{file_name}_{dataset_name}_{dataset_type}_{classifier_type}_"
        + f"{feature_type}_{enrichment_type}_{constants.DATETIME_FORMAT}.png"
    )


def generate_summary_file(
    docs_total,
    number_categories,
    final_accuracy,
    execution_time,
    number_drifts,
    model_summary,
    dataset_name,
    dataset_type,
    file_name,
    classifier_type,
    feature_type,
    enrichment_type,
):
    summary_folder = os.getenv("SUMMARY_FOLDER")
    with open(
        f"{summary_folder}summary_{file_name}_{dataset_name}_{dataset_type}_"
        + f"{classifier_type}_{feature_type}_{enrichment_type}_"
        + f"{constants.DATETIME_FORMAT}.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            ["nº documents", "categories", "mean_accuracy", "time", "drifts", "summary"]
        )
        writer.writerow(
            [
                docs_total,
                number_categories,
                final_accuracy,
                execution_time,
                number_drifts,
                model_summary,
            ]
        )
    file.close()


def generate_aux_plot_file(preq, preq_a, preq_w, plot_aux_file, args):
    df = pd.DataFrame({"preq": preq, "preq_a": preq_a, "preq_w": preq_w})
    df.to_csv(
        f"{plot_aux_file}_{str(args.capitalization)}_{args.classification_type}_"
        + f"{args.feature_extraction}_{args.text}_{args.dataset_type}_plot_aux.csv",
        index=False,
    )


def generate_tree_file(
    model,
    dataset_name,
    dataset_type,
    file_name,
    classifier_type,
    feature_type,
    enrichment_type,
):
    trees_folder = os.getenv("TREES_FOLDER")
    with open(
        f"{trees_folder}tree_{file_name}_{dataset_name}_{dataset_type}_"
        + f"{classifier_type}_{feature_type}_{enrichment_type}_"
        + f"{constants.DATETIME_FORMAT}.dot",
        "w",
    ) as f:
        f.write(str(model.draw()))
