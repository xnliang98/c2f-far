# C2F-FAR
This is the code for COLING 2022 paper: [An Efficient Coarse-to-Fine Facet-Aware Unsupervised Summarization Framework based on Semantic Blocks](https://arxiv.org/abs/2208.08253)

> *Update 2022-08-31: Simple code without tidying up.*

> *TODO: I will refactor the code later when I have time.*

> We resplit the sentence of billsum with stanfordcorenlp v4.5.1 and place the processed file in `data/billsum/test.json`.

## requirements
- python == 3.7.0
- `pip install -r requirements.txt`

## Runing from the scratch
The YAML format config files of all steps are placed in `yaml_config/`.

### Step 1: Obtain the sentence level document embeddings 
Firstly, you should modify the config in `yaml_config/encode_document.yaml`.
- model_name_or_path: the downloaded pre-trained bert model or sbert model.
- batch_size: the total batch size (we encode the document on 4xV100 32G, if you use different device, you should modify it.)

Then, you can run it by:
```
python src/encode_document.py yaml_config/encode_document.yaml
```
After that, you can see the `test.cls.[model_name].[max_length].pt` file in the folder `tmp_file/billsum/sentence_embeddings/`.

### Step 2: Split the document into Semantic Blocks
You can split the document into Semantic Blocks by running:
```
python src/split_document.py
```

After that, you can see the `test.cls.[model_name].[max_length].[beta].pt` file in the folder `tmp_file/billsum/partitions/`.


If you want to change the model, dataset and so on, you can change them in the line [84-89]:

```python
dataset = "billsum"
beta = 1
data_path = "tmp_file/"
model_name = "sbert"
# model_name = "bert-base-uncased"
max_length = 80
```

### Step 3: Rank and Extract the summary sentences
Firstly, you should modify the config in `yaml_config/ranker.yaml`.
- model_path: just given `sbert` or `bert-base-uncased`.

Then rank and extract the summary sentences by:
```
python src/ranker.py
```
After that, you can see the result file `sbert.cls.dot-product.tokens_200.f_0.b_0.0.l1_0.5.l2_0.5.txt` file in the folder `tmp_file/billsum/c2f_results/`.

### Step 4: Evaluate the results
Finally, you can run `python src/metric.py [results_file_path]` to get the results of c2f-far.

## Download the model and embeddings
You can download results, embeddings, and pretrained_models and then place it directly in the `c2f-far/`.

- Google Drive: https://drive.google.com/file/d/1odDgbhXiAYFB7bZxRD-FxvzXHwdTGqGz/view?usp=sharing
- Baidu Cloud: https://pan.baidu.com/s/1PFgH9CQA3DiETWu4Ua-9yw code: `fdik` 



