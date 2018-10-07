from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word
import json, re, os, sys
import nltk
from nltk import pos_tag, sent_tokenize
from nltk import ngrams
from glob import glob
from collections import defaultdict
import sys



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
    train = CoNLLDataset(config.filename_train, processing_word)
    print('Loaded dev, test, train')

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
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
    train = CoNLLDataset(config.filename_train)
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
    id_to_labels = defaultdict(lambda: {})
    id_to_tokens = {}
    id_to_pos = {}
    id_to_hir = {}
    crowd_ids, gold_ids = set(), set()
    model_name = sys.argv[1]
    # print('Reading files for %s' % model_name)
    #
    # test_labels = glob('../../ebm_nlp_1_00/annotations/aggregated/hierarchical_labels/%s/test/gold/*.ann' % model_name)
    # for fname in test_labels:
    #     pmid = fname_to_pmid(fname)
    #     gold_ids.add(fname_to_pmid(fname))
    #
    #
    # crowd_labels = glob('../../ebm_nlp_1_00/annotations/aggregated/hierarchical_labels/%s/train/*.ann' %model_name)
    # #
    # for fname in crowd_labels:
    #     crowd_ids.add(fname_to_pmid(fname))
    # print('processing %d files' %len(crowd_labels + test_labels))
    # for fname in crowd_labels + test_labels:
    #     pmid = fname_to_pmid(fname)
    #     id_to_labels[pmid][model_name] = open(fname).read().split(',')
    #     #print(id_to_labels)
    #     if pmid not in id_to_tokens:
    #       tokens, tags = zip(*nltk.pos_tag(open('../../ebm_nlp_1_00/documents/%s.tokens' %pmid).read().split()))
    #       id_to_tokens[pmid] = tokens
    #       id_to_pos[pmid] = tags
    #
    #
    # crowd_ids = list(filter(lambda pmid: len(id_to_labels[pmid]) == len(PIO), crowd_ids))
    # dev_idx = int(len(crowd_ids) * 0.2)
    # dev_ids, train_ids = crowd_ids[:dev_idx], crowd_ids[dev_idx:]
    #
    # gold_ids = list(filter(lambda pmid: len(id_to_labels[pmid]) == len(PIO), gold_ids))
    # test_ids = gold_ids
    # for ids, fname in [(dev_ids, 'dev'), (train_ids, 'train'), (test_ids, 'test')]:
    #
    #   fout = open('data/%s_%s.txt' %(fname,model_name), 'w')
    #   #print("fout-->", fout)
    #   for pmid in ids:
    #     try:
    #       p_labels = open('../../ebm_nlp_1_00/annotations/aggregated/starting_spans/%s/test/gold/%s_AGGREGATED.ann' %(model_name,pmid)).read().split(',')
    #     except FileNotFoundError:
    #       p_labels = open('../../ebm_nlp_1_00/annotations/aggregated/starting_spans/%s/train/%s_AGGREGATED.ann'%(model_name, pmid)).read().split(',')
    #
    #     fout.write('-DOCSTART- -X- N\n\n')
    #     for i, (token, pos) in enumerate(zip(id_to_tokens[pmid], id_to_pos[pmid])):
    #       labels = [int(id_to_labels[pmid][model_name][i])]
    #       print(labels)
    PIO = ['participants', 'interventions', 'outcomes']
    for pio in PIO:
      print('Reading files for %s' %pio)


      test_labels = glob('../../ebm_nlp_1_00/annotations/aggregated/hierarchical_labels/%s/test/gold/*.ann' %pio)
      print("-------->", len(test_labels))


      for fname in test_labels:
          #pmid = fname_to_pmid(fname)
          gold_ids.add(fname_to_pmid(fname))
      print("len gold ids", len(gold_ids))

      crowd_labels = glob('../../ebm_nlp_1_00/annotations/aggregated/hierarchical_labels/%s/train/*.ann' %pio)

      for fname in crowd_labels:

          crowd_ids.add(fname_to_pmid(fname))
          # else:
          #   print("false")

      print("len crowd ids--->",len(crowd_ids))

      #print(crowd_ids)


      print('processing %d files' %len(crowd_labels + test_labels))
      for fname in crowd_labels + test_labels:
        pmid = fname_to_pmid(fname)
        id_to_labels[pmid][pio] = open(fname).read().split(',')
        #print(id_to_labels)
        if pmid not in id_to_tokens:
          tokens, tags = zip(*nltk.pos_tag(open('../../ebm_nlp_1_00/documents/%s.tokens' %pmid).read().split()))
          id_to_tokens[pmid] = tokens
          id_to_pos[pmid] = tags


    crowd_ids = list(filter(lambda pmid: len(id_to_labels[pmid]) == len(PIO), crowd_ids))
    dev_idx = int(len(crowd_ids) * 0.2)
    dev_ids, train_ids = crowd_ids[:dev_idx], crowd_ids[dev_idx:]

    gold_ids = list(filter(lambda pmid: len(id_to_labels[pmid]) == len(PIO), gold_ids))
    test_ids = gold_ids


    for ids, fname in [(dev_ids, 'dev'), (train_ids, 'train'), (test_ids, 'test')]:
      newpath = r'data/%s' %model_name
      if not os.path.exists(newpath):
          os.makedirs(newpath)
      fout = open('data/%s/%s.txt' %(model_name,fname), 'w')
      #print("fout-->", fout)
      for pmid in ids:
        try:
          p_labels = open('../../ebm_nlp_1_00/annotations/aggregated/starting_spans/participants/test/gold/%s_AGGREGATED.ann' %pmid).read().split(',')
        except FileNotFoundError:
          p_labels = open('../../ebm_nlp_1_00/annotations/aggregated/starting_spans/participants/train/%s_AGGREGATED.ann'%pmid).read().split(',')
        try:
          i_labels = open('../../ebm_nlp_1_00/annotations/aggregated/starting_spans/interventions/test/gold/%s_AGGREGATED.ann' %pmid).read().split(',')
        except FileNotFoundError:
          i_labels = open('../../ebm_nlp_1_00/annotations/aggregated/starting_spans/interventions/train/%s_AGGREGATED.ann'%pmid).read().split(',')
        try:
          o_labels = open('../../ebm_nlp_1_00/annotations/aggregated/starting_spans/outcomes/test/gold/%s_AGGREGATED.ann' %pmid).read().split(',')
        except FileNotFoundError:
          o_labels = open('../../ebm_nlp_1_00/annotations/aggregated/starting_spans/outcomes/train/%s_AGGREGATED.ann'%pmid).read().split(',')

        #print("P labels", p_labels)
        #print(p_labels)
        fout.write('-DOCSTART- -X- N\n\n')
        for i, (token, pos) in enumerate(zip(id_to_tokens[pmid], id_to_pos[pmid])):
          labels = [int(id_to_labels[pmid][pio][i]) for pio in PIO]
          #print(labels)
          if p_labels[i] == '1':
             # print('true p')
              pio_label = 'P'
          elif o_labels[i] == '1':
              pio_label = 'O'
            #  print('true 0')
          elif i_labels[i] == '1':
              pio_label = 'I'
             # print('true i')
          else:
              pio_label = 'N'
             # print('true N')
          # for j in range(len(labels)):
          #     if labels[j]>0:
          #         h_label = labels[j]
          #         print(h_label)
          #         break
          #     else:
          #         h_label = 0
          if model_name == 'participants':
              word_labels = ['None', 'Age', 'Sex', 'Size', 'Condition']
              h_label = word_labels[labels[0]]
              if pio_label == 'P':
                  fout.write('%s %s %s\n' %(token, pos, h_label))

          if model_name == 'interventions':
              word_labels = ['None', 'Surgical', 'Physical', 'Pharmocological', 'Educational', 'Psychological', 'Other', 'Control']
              h_label = word_labels[labels[1]]
              if pio_label == 'I':
                  fout.write('%s %s %s\n' %(token, pos, h_label))
          if model_name == 'outcomes':
              word_labels = ['None', 'Physical', 'Pain', 'Mortality', 'Adverse', 'Mental','Other']
              h_label = word_labels[labels[2]]
              if pio_label == 'O':
                  fout.write('%s %s %s\n' %(token, pos, h_label))
          #if token == '.': fout.write('\n')

if __name__ == "__main__":
      pre_main()
main()
