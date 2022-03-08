Week3 Project
===
The following are the notes on the sub-tasks of W3 project.

##Level 1: Classifying Product Names to Categories

Transformations:
- Punctuations removal
- Lower casing
- English stemming

Parameters:

     python createContentTrainingData.py --min_products 10 --output=datasets/fasttext/producs_preprocessed.txt

Parameters for training the models:

    fasttext supervised -input products.train -output model_products -lr 1.0 -epoch 100 -wordNgrams 2 -minCountLabel 5

Testing:

    fasttext test model_products.bin products.test 
    N       8102
    P@1     0.73
    R@1     0.73

## Level 2: Derive Synonyms from Content

Running skipgrams model tranining:
     
    fasttext skipgram -input titles.txt -output title_model --minCount 20

    Read 0M words
    Number of words:  1269
    Number of labels: 0
    Progress: 100.0% words/sec/thread:   23587 lr:  0.000000 avg.loss:  1.628985 ETA:   0h 0m 0s 

## Level 3: Integrating Synonyms with Search

Training title model:
     
     fasttext skipgram -input titles.txt -output title_model 
     Read 1M words
     Number of words:  8898
     Number of labels: 0
     Progress: 100.0% words/sec/thread:   14932 lr:  0.000000 avg.loss:  1.412956 ETA:   0h 0m 0s

Testing model:

     Query word? laptops
    laps 0.768417
    laptop 0.750745
    notebooks 0.740477
    tops 0.728701
    ibm 0.71659
    atg 0.716323
    15r 0.708476
    lenmar 0.703209
    omnibook 0.694068
    17r 0.688302

    Query word? dell
    inspiron 0.888676
    xps 0.785471
    17r 0.774458
    i570 0.773835
    798bk 0.771554
    7034pbk 0.76741
    9445nbk 0.764047
    1408nbc 0.762812
    565nbk 0.760882
    sx8100 0.759424

    Query word? macbook
    ibook 0.825973
    omnibook 0.810365
    zenbook 0.794665
    bookendz 0.775852
    bookbook 0.761876
    durabook 0.760389
    ultrabook 0.728197
    ebook 0.72038
    powerbook 0.716868
    nextbook 0.712196


Testing synonyms annotation:

     curl -X POST -d '{"name":"dell", "sku":"dell"}' -H"Content-Type: application/json" http://127.0.0.1:5000/documents/annotate
    {
      "name_synonyms": "inspiron"
    }

    curl -X POST -d '{"name":"thinkpad", "sku":"thinkpad"}' -H"Content-Type: application/json" http://127.0.0.1:5000/documents/annotate
    {
      "name_synonyms": "ideapad lenovo"
    }

    curl -X POST -d '{"name":"cellular", "sku":"cellular"}' -H"Content-Type: application/json" http://127.0.0.1:5000/documents/annotate
    {
      "name_synonyms": "cellstar cellstik cellsuit"
    }
