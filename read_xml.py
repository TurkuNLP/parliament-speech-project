# Code for parsing and converting semparl XML files to CSV.
# This works for files between 2000 to 2021,
# parsing older files requires some modification.
# Output is a CSV file with columns id, text, label, speaker, year

# import modules
import csv
import xml.etree.ElementTree as et
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True,
    help='which year of speeches to parse')
args = parser.parse_args()

#initialise empty lists for later
speaker_name = []
speaker_party = []
speech_text = []
speech_year = []

# The year you want to parse from xml to csv
year = args.year

# Parsing the XML file
if 2015 <= year <= 2021:
   xmlparse = et.parse(f'/scratch/project_2006385/otto/data/semparl-speeches-data-master/XML/2015-2021/Speeches_{year}.xml')
elif 2000 <= year < 2015:
   xmlparse = et.parse(f'/scratch/project_2006385/otto/data/semparl-speeches-data-master/XML/2000-2014/Speeches_{year}.xml')
else:
   raise IndexError('Year must be between 2000 and 2021.')

#get the root element
root = xmlparse.getroot()

#initialise empty dictionary for later
party_dict = {}

#get surname, forename and party of participants in document order
for person in root.findall('./teiheader/profileDesc/particDesc/listPerson/person'):
   name = person.attrib['{http://www.w3.org/XML/1998/namespace}id'] #get person name in format firstname_lastname
   try:
      party = person.find('affliation').attrib['ref'] #party is given as attribute of affliation element
      party = party.replace('#party.','') #each party name is preceded by '#party." e.g. '#party.SDP' which we want to remove
      party_dict.update({name : party}) #add key-value pair to dictionary for later retrieval
   except:
      party = 'N/A' #some speakers miss party information, so in that case they are given party label N/A
      party_dict.update({name : party})

interrupted = ''
for segment in root.findall('./TEI/text/body/div/div/u'): #each speech segment is in element 'u'
    text = segment.text #get the text of the speech
    if not isinstance(text,str): #check that the text is actually str, there are some 'nan's in the data
       continue
    if 'ana' in segment.attrib: #if speaker has attribute 'ana', they are chairman, vicechair or secondvicechair. we want to remove them from the data
       continue
    if 'next' in segment.attrib or 'prev' in segment.attrib: #segments with next and prev are interrupted
        interrupted += text #interrupted segments are combined to form one speech
        if 'next' in segment.attrib:
            continue
        else: #if segment has only attribute 'prev' it's the end of the speech
            text = interrupted
            interrupted = '' 
    speaker = segment.attrib['who'] #get the name of the speaker
    speaker = speaker.replace('#','') #remove preceding hashtag from speaker name
    speaker_name.append(speaker) #append name of speaker to a list
    speaker_party.append(party_dict[speaker]) #get party of speaker from dictionary and append to separate list
    speech_text.append(text) #append the text of the speech to a third list
    speech_year.append(year)


#create pandas dataframe where each row contains speech, party label, name of speaker and year
df = pd.DataFrame({'text': speech_text, 'label': speaker_party, 'speaker': speaker_name, 'year': speech_year})

df.loc[df['label'] == 'SDP', 'label'] = 'SD' #change few occurrances of SDP to more commonly used SD
df.loc[df['label'] == 'RKP', 'label'] = 'R' #change few occurrances of RKP to more commonly used R
df.loc[df['label'] == 'SKL', 'label'] = 'KD' #change few occurrances of SKL to the new name of the party KD

#mapping for converting party names to label numbers, because transformer requires numbered labels
label_dict = {'SD': 0, 'KOK': 1, 'KESK': 2, 'VIHR': 3, 'VAS': 4, 'PS': 5, 'R': 6, 'KD': 7}

#do the conversion
for key in label_dict.keys():
    df.loc[df['label'] == key, 'label'] = label_dict[key]
    
values = list(label_dict.values()) #get list of valid label values

#print labels before dropping to check if there is something interesting there
ls = df['label'].to_list()
print('labels in data before dropping invalid values:')
print((set(ls)))

df = df[df["label"].isin(values)] #drop rows with invalid label values, i.e. parties that do not apper in label_dict

#print all labels to check that there are no errors
ls = df['label'].to_list()
print('labels in clean data:')
print(set(ls))

#save dataframe as csv for later use
#index = False does not include the automatically created index in file as this causes problems when loading later
df.to_csv(f'../data/parl_speeches_2000-2021/parl_speeches_{year}.csv', index_label = 'id')
