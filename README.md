### Parlimanent Speech Classifier

This project aims to train a classifier for predicting political afiliation based on speeches held in the Finnish Parliament. This project is part of the Semparl project.The data used to train the classifier comes from Parlamenttisampo. The data is unfortunately not freely available, but I will upload it if allowed later.

The code is divided into three parts. First, you should run the read_xml.py script, to parse the raw xml files into csv files with only the relevant data remaining. Then, run create_datasets.py to furher process the data and prepare it for the classifier. Finally, run either classifier.py, classifier-with-comet.py or classifier-with-comet.ipynb. Comet is a machine learning platform that automatically logs and creates plots from training experiments. To run the scripts '-with-comet', you need to sign up to https://www.comet.com/site/ and fetch your personal API-key.