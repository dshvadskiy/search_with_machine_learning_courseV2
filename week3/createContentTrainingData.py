import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from nltk.stem import SnowballStemmer
import nltk
tokenizer = nltk.RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("english")
import pandas as pd

def transform_name(product_name):
    # IMPLEMENT
    product_name_transform = " ".join([stemmer.stem(token.lower()) for token in tokenizer.tokenize(product_name)])
    return product_name_transform

# Directory for product data
directory = r'data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate

#ds = pd.DataFrame(columns= ['category','name'])
name_dict = {'category' : [],
             'name': []}

print("Writing results to %s" % output_file)

for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        print("Processing %s" % filename)
        f = os.path.join(directory, filename)
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if random.random() > sample_rate:
                continue
            # Check to make sure category name is valid
            if (child.find('name') is not None and child.find('name').text is not None and
                child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                  # Choose last element in categoryPath as the leaf categoryId
                  cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                  # Replace newline chars with spaces so fastText doesn't complain
                  name = child.find('name').text.replace('\n', ' ')
                  #output.write("__label__%s %s\n" % (cat, transform_name(name)))
                  name_dict['category'].append(f'__label__{cat}')
                  name_dict['name'].append( transform_name(name))

ds = pd.DataFrame(name_dict)
print(ds.shape)

ds_new = ds.groupby('category').filter(lambda x: len(x) >= min_products)
print(ds_new.shape)
ds_new = ds_new.sample(frac=sample_rate)

with open(output_file, 'w') as output:
    for idx,row in ds_new.iterrows():
        output.write(f"{row['category']} {row['name']}\n")