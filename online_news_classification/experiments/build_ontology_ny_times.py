import torch
import pandas as pd
import logging
import networkx as nx
import matplotlib.pyplot as plt
import json
import ast
import re

from dotenv import load_dotenv

from online_news_classification.functions import manage_datasets, setup

load_dotenv()

# Function to filter tags that share the prefix before the hyphen
def filter_tags(row):
    try:
        # Ensure the tags column is a valid list
        if isinstance(row["tags"], str):  # If stored as string, convert it
            tags_list = ast.literal_eval(row["tags"])
        elif isinstance(row["tags"], list):
            tags_list = row["tags"]
        else:
            return []  # Return an empty list if tags are not a list
    except (ValueError, SyntaxError):
        return []

    # Extract the category prefix (before any "-")
    category_prefix = row["category"].split('/')[0] if '/' in row["category"] else row["category"]

    # Remove the category prefix and any exact match of category/category (like "sports/sports")
    # Ensure that we only keep tags that start with the category prefix and avoid redundancy
    return [tag for tag in tags_list if tag != category_prefix and tag != f"{category_prefix}/{category_prefix}" and tag.startswith(f"{category_prefix}/")]

# Sample DataFrame

def replace_slash(lst):
    if isinstance(lst, list):
        return [s.replace('/', '.') for s in lst]
    return lst

def convert_categories(ddf):
    # Convert categories and tags into a dictionary
    data = {}
    number_of_subcategories = 0
    subcategories_list = []
    for _, row in ddf.iterrows():
        category = row["category"].strip()
        subcategories = row["filtered_tags"]  # Assuming tags are stored as stringified lists
        number_of_subcategories += 1
        if subcategories != category:
            subcategories_list.append(subcategories)

        if category not in data:
            data[category] = []
        data[category].append(subcategories)
    return data, number_of_subcategories, subcategories_list

def filter_categories(data):
    # Filter subcategories to only include those containing the category name (for uk-news, use 'uk')
    filtered_data = {}
    for category, subcategories in data.items():
        #keyword = "uk" if category == "uk-news" else category
        keyword = category
        filtered_data[category] = [
            sub for sub in set(subcategories)
            if keyword in sub and sub != keyword and sub != f"{keyword}_{keyword}"
        ]
    return filtered_data

def write_to_json(json_name, data):
    # Save filtered data to a JSON file
    with open(json_name, "w") as json_file:
        json.dump(data, json_file, indent=4)

def build_ontology(data, name_ontology, file_ontology):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges based on categories and filtered subcategories
    for category, subcategories in data.items():
        G.add_node(category)  # Add main category node
        for subcategory in subcategories:
            G.add_node(subcategory)
            G.add_edge(category, subcategory)  # Connect category to subcategory

    # Draw the tree
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # Tree-like structure
    except ImportError:
        print("Warning: sudo apt-get install graphviz graphviz-dev not installed, using shell layout instead.")
        pos = nx.shell_layout(G)
    # Layout for better visualization
    plt.figure(figsize=(50, 8))  # Adjust size for large taxonomies
    nx.draw(G, pos, with_labels=True, node_size=500,
            node_color="lightblue", edge_color="gray",
            font_size=8, font_weight="bold", arrows=True)
    plt.title(name_ontology)
    plt.savefig(file_ontology, dpi=300, bbox_inches="tight")

def remove_punctuation(text):
    # Replace punctuation with an empty string
    return re.sub(r'[^\w\s]', '', text)

def main():
    start_time = setup.initialize("build_ontology",
                                  )
    logging.info(start_time)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running into {device}...")

    # Load CSV file into Dask DataFrame
    ddf = pd.read_csv(
        "../datasets/the_ny_times/filtered_enriched_the_ny_times.csv",
        sep=";", index_col=0
    )

    ddf['tokenized_text'] = ddf['tokenized_text'].apply(remove_punctuation)

    logging.info(f"Total size of initial dataset: {len(ddf)}")
    ddf = ddf[~ddf['category'].isin(['new_york', 'opinion', 'week_in_review', 'nyt_now', 'times_insider', 'the_upshot', 'the_learning_network', 'todays_paper', 'obituaries', 'corrections', 'multimediaphotos'])]
    category_counts = ddf['category'].value_counts()
    categories_to_keep = category_counts[category_counts > 1000].index

    # Filter the original DataFrame
    ddf = ddf[ddf['category'].isin(categories_to_keep)]

    # Explode the tags column
    ddf_exploded = ddf.explode('filtered_tags')

    # Count occurrences of each tag
    tag_counts = ddf_exploded['filtered_tags'].value_counts()

    # Keep only tags that appear at least 10 times
    valid_tags = tag_counts[tag_counts >= 250]

    x = 1000 # Change this to your desired number
    top_x_tags = valid_tags.head(x).index
    logging.info(f"Top X tags: {top_x_tags}")

    # Filter the DataFrame
    df_filtered = ddf_exploded[ddf_exploded['filtered_tags'].isin(top_x_tags)]

    # Group back to original format
    df_result = df_filtered.groupby(df_filtered.index)['filtered_tags'].apply(list).reset_index()

    # Merge back with the original DataFrame to retain other columns
    ddf = ddf.drop(columns=['filtered_tags']).merge(df_result, left_index=True, right_on='Index', how='left').drop(
        columns=['Index'])

    # Fill NaN values with empty lists where tags were removed completely
    ddf['filtered_tags'] = ddf['filtered_tags'].apply(lambda x: x if isinstance(x, list) else [])

    logging.info(f"Total size of filtered dataset: {len(ddf)}")

    ddf['filtered_tags'] = ddf['filtered_tags'].apply(lambda x: str(x[0]) if x else "")
    ddf = ddf.fillna("").astype(str)

    ddf['filtered_tags'] = ddf['filtered_tags'].apply(lambda x: f"{x}.{x}" if x != '' and '.' not in x else x)

    midpoint = len(ddf) // 2

    # Split into two DataFrames
    df1 = ddf.iloc[:midpoint]  # First half
    df2 = ddf.iloc[midpoint:]  # Second half

    new_data, number_of_subcategories, subcategories_list = convert_categories(df1)
    logging.info(f"Number of categories: {len(set(ddf['category']))}")
    logging.info(f"Number of subcategories: {len(subcategories_list)}")

    #
    new_filtered_data = filter_categories(new_data)
    #

    write_to_json("ontology_strucuture_with_filtering_the_ny_times.json", new_filtered_data)
    #
    build_ontology(new_filtered_data, "Ontology Structure (with filter categories)",
                   "ontology_strucuture_with_filtering_the_ny_times.png")

    manage_datasets.save_dataset(df1, "/the_ny_times/filtered_enriched_the_ny_times_1.csv")

    manage_datasets.save_dataset(df2, "/the_ny_times/filtered_enriched_the_ny_times_2.csv")


if __name__ == "__main__":
    main()