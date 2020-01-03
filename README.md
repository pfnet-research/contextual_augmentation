# Contextual Augmentation

This repository contains a collection of scripts for an experiment of [Contextual Augmentation](https://arxiv.org/pdf/1805.06201.pdf).

Contextual augmentation is a domain-independent data augmentation for text classification tasks.
Texts in supervised dataset are augmented by replacing words with other words
which are predicted by a label-conditioned bi-directional language model.

Disclaimer: PFN provides no warranty or support for this implementation. Use it at your own risk. See license for details.


## References

Sosuke Kobayashi. *Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations*. NAACL-HLT, 2018. [arxiv](https://arxiv.org/pdf/1805.06201.pdf)


### Prepare a label-conditional bi-directional language model

```
# download wikitext
sh prepare_rawwikitext.sh

# install chainer and spacy
pip install cupy
pip install chainer
pip install progressbar2
pip install spacy
python -m spacy download en_core_web_sm

# segment text by sentence boundaries (very slowly)
PYTHONIOENCODING=utf-8 python preprocess_spacy.py -d datasets/wikitext-103-raw/wiki.train.raw > datasets/wikitext-103-raw/spacy_wikitext-103-raw.train
PYTHONIOENCODING=utf-8 python preprocess_spacy.py -d datasets/wikitext-103-raw/wiki.valid.raw > datasets/wikitext-103-raw/spacy_wikitext-103-raw.valid

# construct vocabulary on wikitext
python construct_vocab.py --data datasets/wikitext-103-raw/spacy_wikitext-103-raw.train -t 50 --save datasets/wikitext-103-raw/spacy_wikitext-103-raw.train.vocab.t50

# train a bi-directional language model
python -u train.py -g 0 --train datasets/wikitext-103-raw/spacy_wikitext-103-raw.train --valid datasets/wikitext-103-raw/spacy_wikitext-103-raw.valid --vocab datasets/wikitext-103-raw/spacy_wikitext-103-raw.train.vocab.t50 -u 1024 --layer 1 --dropout 0.1 --batchsize 64 --out trained_bilm

# construct vocabulary on classification datasets
sh construct_vocab_classification.sh

# modify the model file for a different vocabulary
python alt_model_to_another_vocab.py --vocab datasets/wikitext-103-raw/spacy_wikitext-103-raw.train.vocab.t50 --new-vocab vocabs/stsa.fine.vocab.json -u 1024 --layer 1 --resume trained_bilm/best_model.npz --suffix stsa.fine

# finetune the model as a conditional version
python -u train.py -g 0 --labeled-dataset stsa.fine --vocab vocabs/stsa.fine.vocab.json -u 1024 --layer 1 --dropout 0.3 --batchsize 64 --out trained_bilm/stsa.fine --resume trained_bilm/best_model.npz.stsa.fine --epoch 10 --lr 1e-3
```


### Train a classifier

Baseline
```
python train_text_classifier.py -g 0 --dataset stsa.fine --resume-vocab vocabs/stsa.fine.vocab.json --out results_base --model rnn
```

Using contextual augmentation
```
python train_text_classifier.py -g 0 --dataset stsa.fine --resume-vocab vocabs/stsa.fine.vocab.json --out results_caug --model rnn -bilm trained_bilm/stsa.fine/best_model.npz -bilm-l 1 -bilm-u 1024
```

## License

MIT License. Please see the LICENSE file for details.
