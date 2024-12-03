from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging

load_dotenv()

def update_prequential_metrics(
    val, index, preds, soma, alpha, soma_a, nr_a, wind, soma_w
):
    # Prequential calculation
    soma += val
    preq = soma / (index + 1)

    # Prequential Alpha calculation
    soma_a = val + alpha * soma_a
    nr_a = 1 + alpha * nr_a
    preq_a = soma_a / nr_a

    # Prequential Window calculation
    soma_w += val
    if index >= wind:
        soma_w -= preds[index - wind]
        preq_w = soma_w / wind
    else:
        preq_w = soma_w / (index + 1)

    return preq, preq_a, preq_w, soma, soma_a, nr_a, soma_w


def save_plot_and_results(
    args,
    preq,
    preq_a,
    preq_w,
    index,
):
    plot_file = os.path.join(
        os.getenv("RESULTS_FOLDER") + "/plot/" + args.dataset,
    )
    plot_filename = (
        f"{plot_file}_{args.dataset}_{args.dataset_type}_{str(args.model).split('/')[-1]}_{args.experiment}_plot.png"
    )

    # create plot
    _, ax = plt.subplots(figsize=(40, 20))
    ax.plot(range(index), preq, label="Prequential")
    ax.plot(range(index), preq_a, label="Prequential Alpha")
    ax.plot(range(index), preq_w, label="Prequential Window")
    ax.legend()

    # Salvar o gráfico em ficheiro
    plt.savefig(plot_filename)
    plt.close()

    plot_aux_file = os.path.join(
        os.getenv("RESULTS_FOLDER")  + "/plot_aux/" + args.dataset,
    )
    plot_aux_filename = (
        f"{plot_aux_file}_{args.dataset}_{args.dataset_type}_{str(args.model).split('/')[-1]}_{args.experiment}_plot_aux.csv"
    )
    prequential_df = pd.DataFrame({"preq": preq, "preq_a": preq_a, "preq_w": preq_w})

    prequential_df.to_csv(plot_aux_filename, index=False)