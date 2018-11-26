'''
Creates the following dictionaries to be used further in the pipeline from the data in the ConLL2013 format (the format used for training the model):
 {sent:pmid}
 {sent:[pos tags]}
 {sent:[span labels]}

and the following list (to control for the order):
 [list of sentences]

#TODO: combine the dicts into one

'''

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import pickle

with open('data/train.txt') as file:
    data = file.readlines()
    #data = sent_tokenize(data)
    #for i in data:
    #    print(i,'\n')
sent_list = []
sent_id_dict = {}
sent2pos_dict = {}
sent2span_labels_dict = {}

doc_id = 0
sent = []
labels = []
poses = []
for line in tqdm(data):
    if line.startswith('-DOC'):
        doc_id = word_tokenize(line)[1][1:-1]
        #print(doc_id)
    elif len(line)==1:
        if sent not in sent_list:
          sent_list.append(sent)
        sent_id_dict[' '.join(sent)]=doc_id
        sent2pos_dict[' '.join(sent)] = poses
        sent2span_labels_dict[' '.join(sent)] = labels
        #print(sent)
        #print(poses)
        #print(labels)
        sent = []
        labels = []
        poses = []
	
    else:
        word = word_tokenize(line)[0]
        sent.append(word)
        pos = word_tokenize(line)[1]
        poses.append(pos)
        label = word_tokenize(line)[2]
        labels.append(label)
        #print(sent)
        #print(poses)
        #print(labels)

#print(sent_id_dict)
#print(sent2pos_dict)

pickle.dump(sent_list, open ("sent_list.p", "wb"))
pickle.dump(sent_id_dict, open ("sent_id_dict.p", "wb"))
pickle.dump(sent2pos_dict, open ("sent2pos_dict.p", "wb"))
pickle.dump(sent2span_labels_dict, open ("sent2span_labels_dict.p", "wb"))

print(len(sent_list))
print(len(sent_id_dict))
print(len(sent2pos_dict))
print(len(sent2span_labels_dict))
