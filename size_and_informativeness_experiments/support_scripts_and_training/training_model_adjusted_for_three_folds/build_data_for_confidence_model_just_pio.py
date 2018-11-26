from model.config1 import Config
from model.data_utils1 import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word
import json, re, os, sys
import nltk
from nltk import pos_tag, sent_tokenize
from nltk import ngrams
from glob import glob
from collections import defaultdict
import pickle

def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """

    # get config and processing of words
    # loads PubMeda articles
    config = Config(load=False)
    print('Config')
    processing_word = get_processing_word(lowercase=True)
    print('Processing_word')

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train01 = CoNLLDataset(config.filename_train + '_fold01' + '.txt', processing_word)
    print('Loaded dev, test, train')
    train02 = CoNLLDataset(config.filename_train + '_fold02' + '.txt', processing_word)
    print('Loaded dev, test, train')
    train12 = CoNLLDataset(config.filename_train + '_fold12' + '.txt', processing_word)
    print('Loaded dev, test, train')

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train01, train02,train12, dev, test])
    print('Loading vocab_words')
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train + '.txt')
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)

def index_map(span, indices):
    for i in range(len(indices)):
        if indices[i][0] <= span[0]:
            begin = i
        elif span[0] >= indices[i][0] and span[0] <= indices[i][1]:
            begin = i
        if indices[i][1] >= span[1]:
            end = i
            break
        elif span[1] >= indices[i][0] and span[1] <= indices[i][1]:
            end = i
            break
        if i == len(indices)-1: end = i
    return begin, end+1

def tokenize(s):
    """
    :param s: string of the abstract
    :return: list of word with original positions
    """
    def white_char(c):
        return c.isspace() or c in [',', '?']
    res = []
    i = 0
    while i < len(s):
        while i < len(s) and white_char(s[i]): i += 1
        l = i
        while i < len(s) and (not white_char(s[i])): i += 1
        r = i
        if s[r-1] == '.':       # consider . a token
            res.append( (s[l:r-1], l, r-1) )
            res.append( (s[r-1:r], r-1, r) )
        else:
            res.append((s[l:r], l, r))
    return res

def fname_to_pmid(fname):
    pmid = os.path.splitext(os.path.basename(fname))[0].split('_')[0]
    return pmid

def pre_main():
    '''
    creates dictionaries of the format {fileid:{type:labels}} and writes train, dev, and test files from those dictionaries
    '''
    id_to_labels = defaultdict(lambda: {})
    #id_to_h_labels = defaultdict(lambda: {})
    id_to_tokens = {}
    id_to_pos = {}
    crowd_ids, gold_ids = set(), set()
    PIO = ['participants', 'interventions', 'outcomes']
    #for pio in PIO:
      #print('Reading files for %s' %pio)
    crowd_ids = pickle.load(open ("../../pmids/non_test_span_pmids.p", "rb"))
      # for fname in crowd_labels: crowd_ids.add(fname_to_pmid(fname))
      # print(crowd_ids)

    test_ids = pickle.load(open ("../../pmids/span_test_pmids.p", "rb")) #glob('data/ebm_nlp_1_00/annotations/aggregated/starting_spans/%s/test/gold/*.ann' %pio)
      # for fname in test_labels: gold_ids.add(fname_to_pmid(fname))

     #print('processing %d files' %len(crowd_labels + test_labels))
    #crowd_labels = []
    print("len test ids: ", len(test_ids))
    print("len crowd ids: ", len(crowd_ids))
    for pio in PIO:
        crowd_labels = []
        #crowd_h_labels = []
        #crowd_labels = glob('data/ebm_nlp_1_00/annotations/aggregated/starting_spans/%s/train/*.ann' %pio)
        #creating the file names to be opened later(?)
        for id in crowd_ids:
            crowd_lab = 'data/ebm_nlp_1_00/annotations/aggregated/starting_spans/%s/train/%s_AGGREGATED.ann' %(pio,id)
            #crowd_h_lab = 'data/ebm_nlp_1_00/annotations/aggregated/hierarchical_labels/%s/train/%s_AGGREGATED.ann' %(pio,id)
            crowd_labels.append(crowd_lab)
            #crowd_h_labels.append(crowd_h_lab)
        print("len crowd labels", len(crowd_labels))
        #print("len h crowd labels", len(crowd_h_labels))
        test_labels = []
        #test_h_labels = []
        #print(test_ids)
        for id in test_ids:
            test_lab =  'data/ebm_nlp_1_00/annotations/aggregated/starting_spans/%s/test/gold/%s_AGGREGATED.ann' %(pio,id)
           # test_h_lab = 'data/ebm_nlp_1_00/annotations/aggregated/hierarchical_labels/%s/test/gold/%s_AGGREGATED.ann' %(pio,id)
            test_labels.append(test_lab)
           # test_h_labels.append(test_h_lab)
        #print(test_labels, len(test_labels))
            #print("--->", test_lab)
        # for fname in test_labels: gold_ids.add(fname_to_pmid(fname))
        #     for pmid in crowd_ids + test_ids:
    #print("len cr lab", len(crowd_labels))
    #print("len test lab", len(test_labels))
        #actually opening the files; should do 'try' bc not every file actually exists
       # for fname in crowd_h_labels+test_h_labels:
            #print(fname)
        #    pmid = fname_to_pmid(fname)
         #   try:
          #      #crowd_ids.add(fname_to_pmid(fname))
           #     id_to_h_labels[pmid][pio] = open(fname).read().split(',')
            #except FileNotFoundError:
             #   id_to_h_labels[pmid][pio] = [0]*len(open('data/ebm_nlp_1_00/documents/%s.tokens' %pmid).read().split())
        for fname in crowd_labels+test_labels:
            #print(fname)
            pmid = fname_to_pmid(fname)
            try:
                #crowd_ids.add(fname_to_pmid(fname))
                id_to_labels[pmid][pio] = open(fname).read().split(',')
            except FileNotFoundError:
                id_to_labels[pmid][pio] = [0]*len(open('data/ebm_nlp_1_00/documents/%s.tokens' %pmid).read().split())
            if pmid not in id_to_tokens:
                 tokens, tags = zip(*nltk.pos_tag(open('data/ebm_nlp_1_00/documents/%s.tokens' %pmid).read().split()))
                 id_to_tokens[pmid] = tokens
                 id_to_pos[pmid] = tags

    #print(id_to_labels)
    #for id in id_to_tokens:
    #    print(id_to_tokens[id], id_to_pos[id])
    #print(id_to_pos)
    crowd_ids = list(filter(lambda pmid: len(id_to_labels[pmid]) == len(PIO), crowd_ids))
    dev_idx = int(len(crowd_ids) * 0.2)
    dev_ids, train_ids = crowd_ids[:dev_idx], crowd_ids[dev_idx:]
    #
    fold1_train_idx, fold2_train_idx = int(len(train_ids) * 0.34), int(len(train_ids) * 0.67)
    print("fold indices: ", fold1_train_idx, fold2_train_idx)
    fold1_ids, fold2_ids, fold3_ids = train_ids[:fold1_train_idx], train_ids[fold1_train_idx:fold2_train_idx], train_ids[fold2_train_idx:]
    print("len folds: ", len(fold1_ids), len(fold2_ids), len(fold3_ids))
    #TODO deal with test data somehow -- will be different for every kind... OR is already created?? i eliminated all the test files from training, so should not be a problem/no need to rebuild
    # gold_ids = list(filter(lambda pmid: len(id_to_labels[pmid]) == len(PIO), gold_ids))
    # test_ids = gold_ids
    #
    #for ids, fname in [(dev_ids, 'dev'), (train_ids, 'train')]:
    #train and test -- no need to loop through folds--all the same files
    for ids, fname in [(dev_ids, 'dev'), (train_ids, 'train')]:
      fout = open('data/%s.txt' %fname, 'w')
      for pmid in ids:
        fout.write('-DOCSTART- -%s- O O\n\n' %pmid)
        for i, (token, pos) in enumerate(zip(id_to_tokens[pmid], id_to_pos[pmid])):
          labels = [int(id_to_labels[pmid][pio][i]) for pio in PIO]
          label = 'N'
          for pio, is_applied in zip(PIO, labels):
            if is_applied:
              label = pio[0]
         # h_labels = [int(id_to_h_labels[pmid][pio][i]) for pio in PIO]


          fout.write('%s %s %s\n' %(token, pos, label))
          if token == '.': fout.write('\n')
    #for folds, have to loop through training
    folds = [fold1_ids, fold2_ids, fold3_ids]
    for fold in folds:
        for ids, fname in [(fold, 'train')]:
          fout = open('data/fold%s_%s.txt' %(folds.index(fold),fname), 'w')
          for pmid in ids:
            fout.write('-DOCSTART- -%s- O O\n\n' %pmid)
            for i, (token, pos) in enumerate(zip(id_to_tokens[pmid], id_to_pos[pmid])):
              labels = [int(id_to_labels[pmid][pio][i]) for pio in PIO]
              label = 'N'
              for pio, is_applied in zip(PIO, labels):
                if is_applied:
                  label = pio[0]
         # h_labels = [int(id_to_h_labels[pmid][pio][i]) for pio in PIO]


              fout.write('%s %s %s\n' %(token, pos, label))
              if token == '.': fout.write('\n')

if __name__ == "__main__":
    pre_main()
    main()
