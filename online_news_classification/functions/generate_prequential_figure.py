import sys

import matplotlib.pyplot as plt
import pandas as pd


def ler_csv_em_colunas_com_pandas(filepath):
    df = pd.read_csv(filepath, sep=",", decimal=".")
    colunas = {coluna: df[coluna].tolist() for coluna in df.columns}
    return colunas


def main():
    # Chamar a função passando o caminho do arquivo CSV
    non_enriched_df = ler_csv_em_colunas_com_pandas(str(sys.argv[1]))

    non_enriched_preq = non_enriched_df["preq"]
    non_enriched_preq = [min(num, 0.99) for num in non_enriched_preq]
    non_enriched_preq_a = non_enriched_df["preq_a"]
    non_enriched_preq_a = [min(num, 0.99) for num in non_enriched_preq_a]
    non_enriched_preq_w = non_enriched_df["preq_w"]
    non_enriched_preq_w = [min(num, 0.99) for num in non_enriched_preq_w]

    enriched_df = ler_csv_em_colunas_com_pandas(str(sys.argv[2]))
    enriched_preq = enriched_df["preq"]
    enriched_preq = [min(num, 0.99) for num in enriched_preq]
    enriched_preq_a = enriched_df["preq_a"]
    enriched_preq_a = [min(num, 0.99) for num in enriched_preq_a]

    enriched_preq_w = enriched_df["preq_w"]
    enriched_preq_w = [min(num, 0.99) for num in enriched_preq_w]

    # create prequential
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(range(len(enriched_preq)), non_enriched_preq, label="Non-Enriched")
    ax.plot(range(len(enriched_preq)), enriched_preq, label="Enriched")
    ax.set_ylabel("prequential error rate")
    ax.set_xlabel("# documents")
    ax.legend()
    plt.savefig(
        "graphics/experiment_"
        + str(sys.argv[4])
        + "_"
        + str(sys.argv[3])
        + "_"
        + str(sys.argv[5])
        + "_prequential.png"
    )

    # create prequential alpha
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(range(len(enriched_preq_a)), non_enriched_preq_a, label="Non-Enriched")
    ax.plot(range(len(enriched_preq_a)), enriched_preq_a, label="Enriched")
    ax.set_ylabel("prequential error rate")
    ax.set_xlabel("# documents")
    ax.legend()
    plt.savefig(
        "graphics/experiment_"
        + str(sys.argv[4])
        + "_"
        + str(sys.argv[3])
        + "_"
        + str(sys.argv[5])
        + "_prequential_alpha.png"
    )

    # create prequential window
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(range(len(enriched_preq_w)), non_enriched_preq_w, label="Non-Enriched")
    ax.plot(range(len(enriched_preq_w)), enriched_preq_w, label="Enriched")
    ax.set_ylabel("prequential error rate")
    ax.set_xlabel("# documents")
    ax.legend()
    plt.savefig(
        "graphics/experiment_"
        + str(sys.argv[4])
        + "_"
        + str(sys.argv[3])
        + "_"
        + str(sys.argv[5])
        + "_prequential_window.png"
    )


if __name__ == "__main__":
    main()
