#Project Assessment

To assess your project work, you should be able to answer the following questions:

###For query classification:

- How many unique categories did you see in your rolled up training data when you set the minimum number of queries per category to 
  - 100: 880 categories 
  - 1000: 
  
For 1000 minimum queries: 
- What values did you achieve for 
  - P@1: 0.518  R@1: 0.518
  - P@3: 0.234  R@3: 0.702
  - P@5: 0.153  R@5: 0.767

For 100 minimum queries: 
- What values did you achieve for 
  - P@1: 0.51  R@1: 0.51
  - P@3: 0.23  R@3: 0.69 
  - P@5: 0.15  R@5: 0.751

Training parameters:

       fasttext supervised -input labeled_query_data_train.txt -output model_query_categories -lr 0.3 -epoch 100 -wordNgrams 2

- You should have tried at least a few different models, varying the minimum number of queries per category as well as trying different fastText parameters or query normalization. Report at least 3 of your runs.

###For integrating query classification with search:

- Give 2 or 3 examples of queries where you saw a dramatic positive change in the results because of filtering. Make sure to include the classifier output for those queries.

  - query: charger plug;  label: pcmcat183800050007;  confidence: 0.97
  - query: touchpad;  label: pcmcat209000050008; confidence: 0.91
  - query: ipod;  label: abcat0201011; confidence: 0.62

- Given 2 or 3 examples of queries where filtering hurt the results, either because the classifier was wrong or for some other reason. Again, include the classifier output for those queries.
  - query: iphone cover label: pcmcat242000050002 confidence: 0.52
  - query: notebook;  label: pcmcat247400050000; confidence: 0.23
    - query: notebook;  label: cat02015; confidence: 0.14
    - query: notebook;  label: pcmcat164200050013; confidence: 0.14
  - query: guitar;  label: pcmcat151600050002; confidence: 0.19
    - query: guitar;  label: cat02015; confidence: 0.099
    - query: guitar;  label: abcat0715001; confidence: 0.063
    - query: guitar;  label: pcmcat252700050006; confidence: 0.058