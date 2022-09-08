#this is a script for running the classifier
#based largely on https://github.com/TurkuNLP/Deep_Learning_in_LangTech_course/blob/master/hf_trainer_bert.ipynb

#import required modules
from pprint import PrettyPrinter
pprint = PrettyPrinter(compact=True).pprint

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
import datasets
import transformers
import pandas as pd

#load in train and test data
print('Loading dataset...')
#dataset = load_dataset('csv', data_files={'train': ['../data/parl_speeches_2000.csv', '../data/parl_speeches_2001.csv', '../data/parl_speeches_2002.csv', '../data/parl_speeches_2003.csv', '../data/parl_speeches_2004.csv', '../data/parl_speeches_2005.csv', '../data/parl_speeches_2006.csv'],
#                                              'test': '../data/parl_speeches_2010.csv'})

#dataset = load_dataset('csv', data_files={'train': ['../data/parl_speeches_2011.csv', '../data/parl_speeches_2012.csv', '../data/parl_speeches_2013.csv', '../data/parl_speeches_2014.csv', '../data/parl_speeches_2015.csv', '../data/parl_speeches_2016.csv', '../data/parl_speeches_2017.csv'],
#                                              'test': ['../data/parl_speeches_2020.csv', '../data/parl_speeches_2021.csv']})


#these datasets contain speeches between 2000 and 2001 from the three parties that held the bulk of speehces: KOK, SD, KESK
#the data has been shuffled, i.e., its not in chronological order
dataset = load_dataset('csv', data_files = {'train': '../data/parl_speeches_2000-2021_threeparties_train.csv',
                                                'test': '../data/parl_speeches_2000-2021_threeparties_test.csv'})

#shuffle the dataset for good measure
dataset=dataset.shuffle()

#let's see what the data looks like
print("Here's the dataset:")
pprint(dataset)
print('This is an example sentence from the dataset:')
pprint(dataset["train"][0])

#get the number of labels, which is required for the model
train_labels = dataset['train']['label']
test_labels = dataset['test']['label']
num_labels = len(set(train_labels + test_labels))
print('Here are the labels:')
print(set(train_labels + test_labels))
print('Number of labels:')
print(num_labels)


#initialise model and tokenizer
model_name='TurkuNLP/bert-base-finnish-cased-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a simple function that applies the tokenizer
# maximum length of BERT models is 512 due to the position embeddings
def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=512,
        truncation=True
    )
    
# Apply the tokenizer to the whole dataset using .map()
#batched means that tokenization is done in batched, and hence, faster
print('Tokenizing...')
dataset = dataset.map(tokenize, batched=True)

#print test sample of tokens
example = dataset['train'][0]['text']
print('This is an example sentence tokenized:')
tokenized = tokenizer(example)
print(tokenized)
print('Tokens:')
tokens = tokenizer.tokenize(example)
print(tokens)

#initialise model for sequence classification
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Set training arguments
trainer_args = transformers.TrainingArguments(
    output_dir="../results",
    evaluation_strategy="steps",
    logging_strategy="steps",
    load_best_model_at_end=True,
    eval_steps=100,
    logging_steps=100,
    learning_rate=0.00001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    max_steps=500,
    #label_names = ['SD', 'KOK', 'KESK', 'VIHR', 'VAS', 'PS', 'R', 'KD']
)

#function for computing accuracy
accuracy = datasets.load_metric("accuracy")

def compute_accuracy(outputs_and_labels):
    outputs, labels = outputs_and_labels
    predictions = outputs.argmax(axis=-1) #pick the index of the "winning" label
    return accuracy.compute(predictions=predictions, references=labels)

#data collator pads the input to be of uniform size
data_collator = transformers.DataCollatorWithPadding(tokenizer)

# Argument gives the number of steps of patience before early stopping
early_stopping = transformers.EarlyStoppingCallback(
    early_stopping_patience=5)

from collections import defaultdict

class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)

training_logs = LogSavingCallback()

# print a sample of test labels to see that they are not ordered
print('Sample of test labels:')
print(dataset["test"]["label"][:50])


#train model using arguments defined above
trainer = None
trainer = transformers.Trainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_accuracy,
    data_collator=data_collator,
    tokenizer = tokenizer,
    callbacks=[early_stopping, training_logs]
)

trainer.train()

eval_results = trainer.evaluate(dataset["test"])
pprint(eval_results)
print('Accuracy:', eval_results['eval_accuracy'])

#print a few example predictions

#THIS DOES NOT WORK AT THE MOMENT

label_text = ['SD', 'KOK', 'KESK', 'VIHR', 'VAS', 'PS', 'R', 'KD']

def predict_party(string):
    tokenized = tokenizer(string, return_tensors='pt')
    pred = model(**tokenized)
    pred_idx = pred.logits.detach().numpy().argmax()
    return label_text[pred_idx]
    
example_sentences = [
  dataset['test']['text'][0],
  dataset['test']['text'][10],
  dataset['test']['text'][20]
  ]

#for e in example_sentences:
    print(e, '->', predict_party(e))

#visualise training results

#THIS DOES NOT WORK AT THE MOMENT

import matplotlib.pyplot as plt

def plot(logs, keys, labels):
    values = sum([logs[k] for k in keys], [])
    plt.ylim(max(min(values)-0.1, 0.0), min(max(values)+0.1, 1.0))
    for key, label in zip(keys, labels):    
        plt.plot(logs["epoch"], logs[key], label=label)
    plt.legend()
    plt.show()

#plot(training_logs.logs, ["loss", "eval_loss"], ["Training loss", "Evaluation loss"])

#plot(training_logs.logs, ["eval_accuracy"], ["Evaluation accuracy"])
