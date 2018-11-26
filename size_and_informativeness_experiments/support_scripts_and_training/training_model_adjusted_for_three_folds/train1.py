from model.data_utils1 import CoNLLDataset
from model.ner_model1 import NERModel
from model.base_model import BaseModel
from model.config1 import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    fold_names = ['fold01', 'fold02', 'fold12']

    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    for fold_name in fold_names:

      train = CoNLLDataset("./data/train_%s.txt" % (fold_name), config.processing_word,
                         config.processing_tag, config.max_iter)

      # train model
      print('training')
      model.train(train, dev, fold=fold_name)

if __name__ == "__main__":
    main()
