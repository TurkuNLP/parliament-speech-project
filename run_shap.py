#from pprint import PrettyPrinter
#pprint = PrettyPrinter(compact=True).pprint

from datasets import load_dataset
#import matplotlib.pyplot as plt
import transformers
#import pandas as pd
import numpy as np
import torch
import shap
import pickle

model_name = 'model19_highscores'
model_dir = "../results/models/model19/checkpoint-9700"

model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
cache_dir = '../hf_cache'

#dataset = load_dataset('csv', data_files = {'test': '../data/2000-2021_80-20-split/parl_speeches_2000-2021_eightparties_test_tokenized_filtered.csv'},
#                       cache_dir = cache_dir)

dataset = load_dataset('csv', data_files = {'test': '../results/data_analysis/model19_predictions_highscore.csv'},
                       cache_dir = cache_dir)

pipeline_all_scores = transformers.TextClassificationPipeline(model = model,
                                                              device = 0,
                                                              batch_size = 32,
                                                              tokenizer = tokenizer,
                                                              truncation = True,
                                                              padding = True,
                                                              max_length = 512,
                                                              return_all_scores=True, #returns all scores, not just winning label
                                                              )

explainer = shap.Explainer(pipeline_all_scores, seed=1234, output_names= ['SDP', 'KOK', 'KESK', 'VIHR', 'VAS', 'PS', 'R', 'KD'])
shap = explainer(dataset['test']['text'])

# Save explainer and shap values.
with open(f'../results/shap_values/explainer_{model_name}.sav',"wb") as f:
    pickle.dump(explainer, f)

with open(f'../results/shap_values/shapvalues_{model_name}.sav',"ab") as f:
    pickle.dump(shap, f)
