""" This is a script for running the classifier without comet-ml.
It is based on https://github.com/TurkuNLP/Deep_Learning_in_LangTech_course/blob/master/hf_trainer_bert.ipynb
To run from command line, give required argument 'xp_name'.
Optional arguments are for changing hyperparameters. Defaults should give decent training results.
"""

#import required modules
from pprint import PrettyPrinter
pprint = PrettyPrinter(compact=True).pprint
import transformers
from datasets import load_dataset
import datasets
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(
            description='A script for predicting the political affiliation of the politician who gave a speech in Finnish parliament'
        )
parser.add_argument('--xp_name', required=True,
    help='give a name for this experiment')
parser.add_argument('--party_num', type=int, default=3,
    help='how many parties should the data contain (3 or 8)')
parser.add_argument('--learning_rate', type=float, default=0.00001,
    help="set trainer learning rate")
parser.add_argument('--batch_size', type=int, default=32,
    help='set trainer batch size')
parser.add_argument('--max_steps', type=int, default=5000,
    help='set trainer max steps')
parser.add_argument('--label_smoothing', type=float, default=0.1,
    help='set label smoothing factor')
args = parser.parse_args()

def main():
    # Setup data paths
    if args.party_num == 3:
        party_num = 'three'
    elif args.party_num == 8:
        party_num = 'eight'
    train_data = f'../data/parl_speeches_2000-2021_{party_num}parties_train.csv'
    validation_data = f'../data/parl_speeches_2000-2021_{party_num}parties_validation.csv'
    test_data = f'../data/parl_speeches_2000-2021_{party_num}parties_test.csv'
    
    cache_dir = '../hf_cache' # hf cache can get bloated with multiple runs so save to disk with enough storage
    output_dir = f'../results/models/model-with-comet/{args.xp_name}' # Where results are saved
    
    # Load in train and test data
    print('Loading dataset...')
    # Speeches between 2000 and 2001 from the three parties that held the bulk of speehces: KOK, SD, KESK
    # The data has been shuffled, i.e., its not in chronological order
    dataset = load_dataset('csv', data_files = {'train': train_data,
                                                'validation': validation_data,
                                                'test': test_data},
                                                cache_dir = cache_dir)
    
    # Shuffle the dataset for good measure
    dataset=dataset.shuffle()

    #let's see what the data looks like
    print("Here's the dataset:")
    pprint(dataset)
    print('This is an example sentence from the dataset:')
    pprint(dataset["train"][0])

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
        id2label_in_data = []
        for i in label_ints:
            id2label_in_data.append(id2label[i])
        label2id_in_data = {}
        for key, value in id2label_in_data:
            label2id_in_data[value] = key
        return num_labels, id2label_in_data, label2id_in_data
    
    num_labels, id2label_in_data, label2id_in_data = get_labels(dataset)


    #initialise model and tokenizer
    model_name='TurkuNLP/bert-base-finnish-cased-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define a simple function that applies the tokenizer
    # maximum length of BERT models is 512 due to the position embeddings
    def tokenize(example):
        return tokenizer(
            example['text'],
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
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                            num_labels = num_labels, 
                                                                            id2label = id2label_in_data,
                                                                            label2id = label2id_in_data)

    # Set training arguments
    trainer_args = transformers.TrainingArguments(
        output_dir = output_dir,
        save_total_limit = 1, #only keep the best model in the end
        evaluation_strategy = 'steps',
        logging_strategy = 'steps',
        load_best_model_at_end = True,
        eval_steps = 100,
        logging_steps = 100,
        learning_rate = args.learning_rate,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        max_steps = args.max_steps,
        label_smoothing_factor = args.label_smoothing
    )

    #function for computing metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average = 'macro'
        )
        acc = accuracy_score(labels, preds)
    
        return {'accuracy': acc, 'macro-f1': f1, 'precision': precision, 'recall': recall}

    #data collator pads the input to be of uniform size
    data_collator = transformers.DataCollatorWithPadding(tokenizer)

    # Argument gives the number of steps of patience before early stopping
    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience = 10)

    # Print a sample of test and validation labels to see that they are not ordered
    print('Sample of test and validation labels:')
    print(dataset['test']['label'][:20])
    print(dataset['validation']['label'][:20])
    

    #train model using arguments defined above
    trainer = None
    trainer = transformers.Trainer(
        model = model,
        args = trainer_args,
        train_dataset = dataset['train'],
        eval_dataset = dataset['validation'],
        compute_metrics = compute_metrics,
        data_collator = data_collator,
        tokenizer = tokenizer,
        callbacks = [early_stopping, training_logs]
    )

    trainer.train()

    # Evaluate results on test set
    # Metrics saved to file for later inspection
    def predict(name):
        pred_results = trainer.predict(dataset["test"])
        with open(f'../results/models/evaluation_{name}.txt', 'w') as f:
            f.write('Accuracy: ')
            f.write(f'{pred_results[2]["test_accuracy"]}\n')
            f.write('Macro-F1: ')
            f.write(f'{pred_results[2]["test_macro-f1"]}\n')
            f.write('Loss: ')
            f.write(f'{pred_results[2]["test_loss"]}\n')

    predict(args.xp_name)

    #print a few example predictions
    model.to('cpu')

    examples = dataset['test'].select(range(5))

    pipeline = transformers.TextClassificationPipeline(
        model = model,
        tokenizer = tokenizer,
        truncation = True,
        max_length = 512
    )
    
    for i in range(len(examples)):
        print('speech:', examples['text'][i])
        print('predicted label:', pipeline(examples['text'][i])[0])
        print('true label:', id2label[examples['label'][i]])
        print('')

    # Let's get the predictions and confidence and add them to a table for later inspection
    model.to('cuda')
    test_pred = trainer.predict(dataset['test'])
    predictions = test_pred.predictions
    pred_labels = predictions.argmax(-1)
    pred_confidence = predictions.max(-1)
    test_data['prediction'] = pred_labels
    test_data['pred_confidence'] = pred_confidence

    test_data.to_csv(f'../results/tables/predictions_{xp_name}.csv', index = False)

if __name__ == '__main__':
    main()