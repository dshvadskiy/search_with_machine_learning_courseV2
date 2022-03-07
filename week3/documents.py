#
# A simple endpoint that can receive documents from an external source, mark them up and return them.  This can be useful
# for hooking in callback functions during indexing to do smarter things like classification
#
from flask import (
    Blueprint, request, abort, current_app, jsonify
)
import fasttext
import json
import nltk
tokenizer = nltk.RegexpTokenizer(r"\w+")

bp = Blueprint('documents', __name__, url_prefix='/documents')

# Take in a JSON document and return a JSON document
@bp.route('/annotate', methods=['POST'])
def annotate():
    if request.mimetype == 'application/json':
        the_doc = request.get_json()
        response = {}
        cat_model = current_app.config.get("cat_model", None) # see if we have a category model
        syns_model = current_app.config.get("syns_model", None) # see if we have a synonyms/analogies model
        # We have a map of fields to annotate.  Do POS, NER on each of them
        sku = the_doc["sku"]
        for item in the_doc:
            the_text = the_doc[item]
            if the_text is not None and the_text.find("%{") == -1:
                if item == "name":
                    if syns_model is not None:
                        nn_k = current_app.config["syns_model_nn_k"]
                        nn_threshold = current_app.config["syns_model_nn_threshold"]
                        analyzed_text = " ".join([token.lower() for token in tokenizer.tokenize(the_text)])
                        #print("IMPLEMENT ME: call nearest_neighbors on your syn model and return it as `name_synonyms`")
                        neighbors = syns_model.get_nearest_neighbors(analyzed_text, k=nn_k)
                        syn_list = []
                        for neighbor in neighbors:
                            neighbor_score = neighbor[0]
                            neighbor_text = neighbor[1]
                            if neighbor_score >= nn_threshold:
                                print(f"\t{neighbor_text} ({neighbor_score}) [>= threshold {nn_threshold}]")
                                syn_list.append(neighbor_text)
                            else:
                                print(f"\t{neighbor_text} ({neighbor_score})")
                        syns_text = ' '.join(syn_list)
                        print(f"\tsyns_text={syns_text}")
                        response[f'{item}_synonyms'] = syns_text
        return jsonify(response)
    abort(415)
