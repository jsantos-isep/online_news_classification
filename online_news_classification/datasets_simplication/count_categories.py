import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

from online_news_classification.functions import setup
from collections import Counter
import logging

load_dotenv()

def main():
    args = setup.get_arg_parser_count_categories().parse_args()
    #start_time = setup.initialize(
    #    "count_categories_" + args.dataset
    #)

    categories = []
    for chunk in pd.read_csv(args.input_file, sep=";", chunksize=1000):
        categories = categories + chunk['category'].tolist()

    frequency_category = Counter(categories)
    logging.info(frequency_category)

    elements = list(frequency_category.keys())
    counts = list(frequency_category.values())

    # Set dynamic width based on number of elements (scale factor)
    # num_elements = len(elements)
    # width = max(10, num_elements * 0.5)  # Ensure a minimum width, but scale based on number of elements
    # Set the figure size
    # plt.figure(figsize=(width, 12))
    #
    # # Create a bar plot
    # plt.bar(elements, counts, color='grey')
    #
    # # Rotate x-axis labels vertically
    # plt.xticks(rotation=90)
    #
    # # Add labels and title
    # plt.xlabel('Categories')
    # plt.ylabel('Frequency')
    # plt.title('Category distribution in ' + args.dataset)
    #
    # for i in range(len(elements)):
    #     plt.text(elements[i], counts[i] + 0.05, str(counts[i]), ha='center')
    #
    # # Save the plot to a file
    # plt.savefig(os.getenv("CATEGORIES_FOLDER") + args.dataset+"_categories_distribution"'.png')  # You can change 'plot.png' to any other file name and format

    # logging.info(len(list(set(categories))))
    print(list(set(categories)))


if __name__ == "__main__":
    main()
