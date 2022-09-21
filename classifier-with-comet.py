#this is a script for running the classifier
#based largely on https://github.com/TurkuNLP/Deep_Learning_in_LangTech_course/blob/master/hf_trainer_bert.ipynb


import argparse

parser = argparse.ArgumentParser(
            description="A script for predicting the political affiliation of the politician who spoke a given speech in Finnish parliament"
        )
parser.add_argument('--xp_name', required=True,
    help = 'give a name for this experiment')
parser.add_argument('--comet_key', required=True,
    help = 'give your comet api key'
parser.add_argument('--comet_workspace', required=True,
    help="give the name of your comet workspace")
parser.add_argument('comet_project', required=True,
    help="give the name of you comet project")
parser.add_argument('--learning_rate', type=float, default=0.00001,
    help="set trainer learning rate")
args = parser.parse_args()


# import comet_ml for logging and plotting
#comet_ml must be imported before any other ML framework for it to work properly
from comet_ml import Experiment, Optimizer
import comet_ml

#import rest of required modules
from pprint import PrettyPrinter
pprint = PrettyPrinter(compact=True).pprint
from datasets import load_dataset
import datasets
import transformers
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
    
def main():    
    
    # Create an experiment with your api key
    def comet(key, space, project):
        experiment = Experiment(
        api_key = key,
        workspace = space,
        project_name = project
        )
        return experiment
    
    experiment = comet(api_key = args.comet_key, args.comet_workspace, args.comet_project)
    
    #setup data paths
    train_data = '../data/parl_speeches_2000-2021_threeparties_train.csv'
    validation_data = '../data/parl_speeches_2000-2021_threeparties_validation.csv'
    test_data = '../data/parl_speeches_2000-2021_threeparties_test.csv'
    
    cache_dir = '../hf_cache' #huggingface cache can get bloated with multiple runs so save to disk with enough storage
    output_dir = f'../results/models/model-with-comet/{args.xp_name}' #where results are saved
    
    #load in train and test data
    print('Loading dataset...')
    #these datasets contain speeches between 2000 and 2001 from the three parties that held the bulk of speehces: KOK, SD, KESK
    #the data has been shuffled, i.e., its not in chronological order
    dataset = load_dataset('csv', data_files = {'train': 'train_data',
                                                'validation' : 'validation_data',
                                                'test': 'test_data'},
                                                cache_dir = cache_dir)
    
    #shuffle the dataset for good measure
    dataset=dataset.shuffle()
    
    #let's see what the data looks like
    print("Here's the dataset:")
    pprint(dataset)
    print('This is an example sentence from the dataset:')
    pprint(dataset["train"][0])
    
    #get the number of labels, which is required for the model
    def get_labels(dataset):
        train_labels = dataset['train']['label']
        label_ints = sorted(list(set(train_labels)))
        num_labels = len(set(train_labels))
        print('Here are the labels:')
        print(label_ints)
        print('Number of labels:')
        print(num_labels)
        id2label = {0 : 'SD', 1 : 'KOK', 2 : 'KESK', 3 : 'VIHR', 4 : 'VAS', 5 : 'PS', 6 : 'R', 7 : 'KD'}
        id2label_in_data = []
        for i in label_ints:
            id2label_in_data.append(id2label[i])
        return num_labels, id2label_in_data
    
    num_labels, id2label_in_data = get_labels(dataset)
    
    #initialise model and tokenizer
    model_name='TurkuNLP/bert-base-finnish-cased-v1'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
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
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, id2label = id2label_in_data)
    
    # Set training arguments
    trainer_args = transformers.TrainingArguments(
        output_dir=output_dir,
        save_total_limit = 1, #only keep the best model in the end
        evaluation_strategy="steps",
        logging_strategy="steps",
        load_best_model_at_end=True,
        eval_steps=100,
        logging_steps=100,
        learning_rate=0.00001,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        max_steps=5000,
        label_smoothing_factor=0.1
    )
    
    #function for computing accuracy and F score
    def get_example(index):
        return dataset['validation'][index]["text"]
    
    
    def compute_metrics(pred):
        experiment = comet_ml.get_global_experiment()
    
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        acc = accuracy_score(labels, preds)
    
        if experiment:
            step = int(experiment.curr_step) if experiment.curr_step is not None else 0
            experiment.set_step(step)
            experiment.log_confusion_matrix(
                y_true=labels,
                y_predicted=preds,
                file_name=f"confusion-matrix-step-{step}.json",
                labels=id2label_in_data,
                index_to_example_function=get_example,
            )
    
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
    #data collator pads the input to be of uniform size
    data_collator = transformers.DataCollatorWithPadding(tokenizer)
    
    # Argument gives the number of steps of patience before early stopping
    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience=5)
    
    # print a sample of test and validation labels to see that they are not ordered
    print('Sample of test and validation labels:')
    print(dataset["test"]["label"][:20])
    print(dataset["validation"]["label"][:20])
    
    
    #train model
    
    #comet_ml automatically logs training data
    os.environ["COMET_LOG_ASSETS"] = "True"
    
    trainer = None
    trainer = transformers.Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer = tokenizer,
        callbacks=[early_stopping]
    )
    
    trainer.train()
    
    def eval:
        eval_results = trainer.evaluate(dataset["test"])
        with open(f'../results/evaluation_{args.xp_name}.txt', 'w') as f:
            f.write('Accuracy: ')
            f.write(f'{eval_results["eval_accuracy"]}\n')
            f.write('F1: ')
            f.write(f'{eval_results["eval_f1"]}\n')
            f.write('Loss: ')
            f.write(f'{eval_results["eval_loss"]}\n')

if __name__ == "__main__":
    main()
