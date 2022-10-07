""" This is a script for running the classifier.
It is based on https://github.com/TurkuNLP/Deep_Learning_in_LangTech_course/blob/master/hf_trainer_bert.ipynb
Running this script requires a comet api key. Read more: https://www.comet.com/site/
To run from command line, give required args 'xp_name', 'comet_key', 'comet_workspace', and 'comet_project'.
Optional arguments are for changing hyperparameters. Defaults should give decent training results.
"""

# Import comet_ml for logging and plotting
# comet_ml must be imported before any other ML framework for it to work properly
from comet_ml import Experiment
import comet_ml

from pprint import PrettyPrinter
pprint = PrettyPrinter(compact = True).pprint
from datasets import load_dataset
import datasets
import transformers
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import argparse

parser = argparse.ArgumentParser(
            description='A script for predicting the political affiliation of the politician who gave a speech in Finnish parliament'
        )
parser.add_argument('--xp_name', required=True,
    help='give a name for this experiment')
parser.add_argument('--comet_key', required=True,
    help='give your comet api key')
parser.add_argument('--comet_workspace', required=True,
    help="give the name of your comet workspace")
parser.add_argument('--comet_project', required=True,
    help="give the name of you comet project")
parser.add_argument('--party_num', type=int, default=8,
    help='how many parties should the data contain (3 or 8)')
parser.add_argument('--train_years', type=str, default=2000, nargs='+',
    help='years of training data, can give more than one separated by comma')
parser.add_argument('--test_years', type=str, default=2019, nargs='+',
    help='years of test data, can give more than one separated by comma')
parser.add_argument('--learning_rate', type=float, default=0.00001,
    help='set trainer learning rate')
parser.add_argument('--batch_size', type=int, default=32,
    help='set trainer batch size')
parser.add_argument('--max_steps', type=int, default=10000,
    help='set trainer max steps')
parser.add_argument('--label_smoothing', type=float, default=0.1,
    help='set label smoothing factor')
args = parser.parse_args()
    
def main(args):    
    
    # Create an experiment with your api key
    def comet(key, space, project):
        experiment = Experiment(
        api_key = key,
        workspace = space,
        project_name = project
        )
        return experiment
    
    experiment = comet(args.comet_key, args.comet_workspace, args.comet_project)
    
    def get_data(party_num, train_years, test_years):
        if party_num == 3:
            party_num = 'three'
        elif party_num == 8:
            party_num = 'eight'
        train_data = []
        test_data = []
        for year in train_years:
            train_data.append(f'../data/parl_speeches_2000-2021/parl_speeches_{year}.csv')
        for year in test_years:
            test_data.append(f'../data/parl_speeches_2000-2021/parl_speeches_{year}.csv')
        cache_dir = '../hf_cache' # hf cache can get bloated with multiple runs so save to disk with enough storage
        #output_dir = f'../results/models/{args.xp_name}' # Where results are saved
        print('Loading dataset...')
        dataset = load_dataset('csv', data_files = {'train': train_data,
                                                    'test': test_data},
                                                    cache_dir = cache_dir)
        dataset=dataset.shuffle()
        return dataset
    
    train_years = [i for i in args.train_years]
    test_years = [i for i in args.test_years]
    print(train_years)
    print(test_years)
    dataset = get_data(args.party_num, train_years, test_years)
    
    # Let's see what the data looks like
    print('Here is the dataset:')
    pprint(dataset)
    print('This is an example sentence from the dataset:')
    pprint(dataset['train'][0])
    
    # Get the number of labels, which is required for the model
    def get_labels(dataset):
        train_labels = dataset['train']['label']
        label_ints = sorted(list(set(train_labels)))
        num_labels = len(set(train_labels))
        print('Here are the labels:')
        print(label_ints)
        print('Number of labels:')
        print(num_labels)
        id2label = {0: 'SD', 1: 'KOK', 2: 'KESK', 3: 'VIHR', 4: 'VAS', 5: 'PS', 6: 'R', 7: 'KD'}
        id2label_in_data = {}
        for key, value in id2label.items():
            if key in label_ints:
                id2label_in_data[key] = value
        label2id_in_data = {}
        for key, value in id2label_in_data.items():
            label2id_in_data[value] = key
        return num_labels, id2label_in_data, label2id_in_data
    
    num_labels, id2label_in_data, label2id_in_data = get_labels(dataset)
    
    print('num labels:', num_labels)
    print('id2label_in_data:', id2label_in_data)
    print('label2id_in_data:', label2id_in_data)

    # Initialise model and tokenizer
    model_name = 'TurkuNLP/bert-base-finnish-cased-v1'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # Define a simple function that applies the tokenizer
    # Maximum length of BERT models is 512 due to the position embeddings
    def tokenize(example):
        return tokenizer(
            example['text'],
            max_length=512,
            truncation=True
        )
        
    # Apply the tokenizer to the whole dataset using .map()
    # Batched means that tokenization is done in batches, and hence, faster
    print('Tokenizing...')
    dataset = dataset.map(tokenize, batched=True)
    
    # Print test sample of tokens
    example = dataset['train'][0]['text']
    print('This is an example sentence tokenized:')
    tokenized = tokenizer(example)
    print(tokenized)
    print('Tokens:')
    tokens = tokenizer.tokenize(example)
    print(tokens)
    
    def filter_short(dataset):
        # remove speeches that are 12 tokens long or shorter
        #too_short = dataset.filter(lambda x: len(x['input_ids']) <= 12)
        dataset = dataset.filter(lambda x: len(x['input_ids']) > 12)
        return dataset
    
    dataset = filter_short(dataset)
    
    # Iinitialise model for sequence classification
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                            num_labels = num_labels, 
                                                                            id2label = id2label_in_data,
                                                                            label2id = label2id_in_data)
    
    # Set training arguments
    trainer_args = transformers.TrainingArguments(
        output_dir = f'../results/models/{args.xp_name}',
        save_total_limit = 1, #only keep the best model in the end
        evaluation_strategy = 'steps',
        logging_strategy = 'steps',
        load_best_model_at_end = True,
        eval_steps = 100,
        logging_steps = 100,
        save_steps = 100,
        metric_for_best_model = 'eval_macro-f1',
        learning_rate = args.learning_rate,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        max_steps = args.max_steps,
        label_smoothing_factor = args.label_smoothing
    )
    
    # Function for computing accuracy and F score
    def get_example(index):
        return dataset['test'][index]['text']
    
    def compute_metrics(pred):
        experiment = comet_ml.get_global_experiment()
    
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average = 'macro'
        )
        acc = accuracy_score(labels, preds)
    
        if experiment:
            step = int(experiment.curr_step) if experiment.curr_step is not None else 0
            experiment.set_step(step)
            experiment.log_confusion_matrix(
                y_true = labels,
                y_predicted = preds,
                file_name = f'confusion-matrix-step-{step}.json',
                labels = list(id2label_in_data.values()),
                index_to_example_function = get_example,
            )
    
        return {'accuracy': acc, 'macro-f1': f1, 'precision': precision, 'recall': recall}
    
    # Data collator pads the input to be of uniform size
    data_collator = transformers.DataCollatorWithPadding(tokenizer)
    
    # Argument gives the number of steps of patience before early stopping
    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience = 10)
    
    # Print a sample of test labels to see that they are not ordered
    print('Sample of test labels:')
    print(dataset['test']['label'][:20])
    
    # Train model
    # comet_ml automatically logs training data
    os.environ['COMET_LOG_ASSETS'] = 'True'
    
    trainer = None
    trainer = transformers.Trainer(
        model = model,
        args = trainer_args,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        compute_metrics = compute_metrics,
        data_collator = data_collator,
        tokenizer = tokenizer,
        callbacks =[early_stopping]
    )
    
    trainer.train()
    
    # Evaluate results on test set
    # Metrics saved to file for later inspection
    def predict(name):
        pred_results = trainer.predict(dataset["test"])
        with open(f'../results/models/evaluation_{name}.txt', 'w') as f:
            f.write('Accuracy: ')
            f.write(f'{pred_results[2]["test_accuracy"]}\n')
            f.write('Macro-f1: ')
            f.write(f'{pred_results[2]["test_macro-f1"]}\n')
            f.write('Loss: ')
            f.write(f'{pred_results[2]["test_loss"]}\n')

    predict(args.xp_name)

if __name__ == '__main__':
    main(args)
