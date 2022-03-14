import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
from cleantext import clean

# Useful if you want to perform stemming.
import nltk

stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'../datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'../datasets/train.csv'
output_file_name = r'../datasets/labeled_query_data.txt'
output_train_file_name = r'../datasets/labeled_query_data_train.txt'
output_test_file_name = r'../datasets/labeled_query_data_test.txt'

sample_size = 50000

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1, help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output
min_queries = args.min_queries


def clean_text(text: str) -> str:
    cleaned_text = clean(text=text, lower=True, no_emoji=True, no_urls=True, no_punct=True, no_line_breaks=True,
                         no_currency_symbols=True, lang="en")
    return " ".join([stemmer.stem(token) for token in cleaned_text.split()])


def get_parent_cat(parent_df, cat):
    if cat == root_category_id:
        return root_category_id
    return parent_df[parent_df['category'] == cat]['parent'].values[0]


def rollup_query_categories(dataset, parent_cat_df, min_queries):
    dataset['cat_size'] = dataset.groupby('category')['query'].transform('count')
    condition = lambda cutoff: dataset['cat_size'] < min_queries

    while len(dataset[condition(min_queries)]):
        dataset.loc[condition(min_queries), ['category']] = dataset.loc[condition(min_queries), ['category']].applymap(
            lambda cat: get_parent_cat(parent_cat_df, cat))
        dataset['cat_size'] = dataset.groupby('category')['query'].transform('count')

    return dataset


if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns=['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

print(f'dataset size before rollup {len(df)}')
print(f'number of unique categories before rollup: {df["category"].nunique()}')

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
df['query'] = df['query'].apply(lambda q: clean_text(q))
# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
df = rollup_query_categories(df, parents_df, min_queries)
print(f'dataset size after rollup {len(df)}')
print(f'number of unique categories after rollup: {df["category"].nunique()}')

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)

df = df.sample(frac=1).reset_index(drop=True)
print('top train categories')
print(df.head(sample_size)['category'].value_counts().head(20))
print('top test categories')
print(df.tail(sample_size)['category'].value_counts().head(20))
df[['output']].head(sample_size).to_csv(output_train_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
df[['output']].tail(sample_size).to_csv(output_test_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)