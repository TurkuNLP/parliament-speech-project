# Code for parsing and converting semparl XML files to CSV

import csv
import xml.etree.ElementTree as et
import pandas as pd
import re

# Initialise empty lists for later
speaker_name = []
speaker_party = []
speech_text = []

# Insert here the year you want to parse from xml to csv
year = '2021'

# Parsing the XML file
xmlparse = et.parse(f'/scratch/project_2006385/otto/data/semparl-speeches-data-master/XML/2015-2021/Speeches_{year}.xml')

# Get the root element
root = xmlparse.getroot()

# Initialise empty dictionary for later
party_dict = {}

# Get name and party of participants in document order
for person in root.findall('./teiheader/profileDesc/particDesc/listPerson/person'):
   name = person.attrib['{http://www.w3.org/XML/1998/namespace}id'] # Get person name in format firstname_lastname
   try:
      party = person.find('affliation').attrib['ref'] # Party is given as attribute of affliation element
      party = party.replace('#party.','') # Each party name is preceded by '#party." e.g. '#party.SDP' which we want to remove
      party_dict.update({name : party}) # Add key-value pair to dictionary for later retrieval
   except:
      party = 'N/A' # Some speakers miss party information, so in that case they are given party label N/A
      party_dict.update({name : party})

for segment in root.findall('./TEI/text/body/div/div/u'): # Each speech segment is in element 'u'
   text = segment.text # Get the text of the speech
   if not isinstance(text,str): # Check that the text is actually str, there are some 'nan's in the data
       continue
   speaker = segment.attrib['who'] # Get the name of the speaker
   speaker = speaker.replace('#','') # Remove preceding hashtag from speaker name
   speaker_name.append(speaker) # Append name of speaker to a list
   speaker_party.append(party_dict[speaker]) # Get party of speaker from dictionary and append to separate list
   speech_text.append(text) # Append the text of the speech to a third list


# Create pandas dataframe where each row contains name of speaker, party of speaker and speech text
df = pd.DataFrame({'text' : speech_text, 'label' : speaker_party})

df.loc[df['label'] == 'SDP', 'label'] = 'SD' # Change few occurrances of SDP to more commonly used SD
df.loc[df['label'] == 'RKP', 'label'] = 'R' # Change few occurrances of RKP to more commonly used R
df.loc[df['label'] == 'SKL', 'label'] = 'KD' # Change few occurrances of SKL to the new name of the party KD

# Mapping for converting party names to label numbers, because transformer requires numbered labels
label_dict = {'SD' : 0, 'KOK' : 1, 'KESK' : 2, 'VIHR' : 3, 'VAS' : 4, 'PS' : 5, 'R' : 6, 'KD' : 7}

# Do the conversion
for key in label_dict.keys():
    df.loc[df['label'] == key, 'label'] = label_dict[key]

# Get list of valid label values
# We only want to keep the 8 parties listed above
values = list(label_dict.values())

# Print labels before dropping to check if there is something interesting there
ls = df['label'].to_list()
print('labels in data before dropping invalid values:')
print((set(ls)))

# Drop rows with invalid label values, i.e. parties that do not apper in label_dict
df = df[df['label'].isin(values)] 

# Print all labels to check that there are no errors
ls = df['label'].to_list()
print('labels in clean data:')
print(set(ls))

# Save dataframe as csv for later use
# index = False removes index from table file as this causes problems when loading later
df.to_csv(f'../data/parl_speeches_{year}.csv', index = False)
