import argparse
import io
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', '-d', required=True)
args = parser.parse_args()

import spacy
nlp = spacy.load('en_core_web_sm', device=2)
nlp.remove_pipe('ner')

text = []
for i_line, l in enumerate(io.open(args.data, encoding='utf8')):
    if i_line % 100000 == 0:
        sys.stderr.write('{} lines end\n'.format(i_line))
    l = l.strip()
    if not l:
        print('')
        continue

    doc = nlp(l)
    tokens = [tok.orth_ for tok in doc]
    for sent in doc.sents:
        print(' '.join(tok for tok in tokens[sent.start: sent.end]
                       if tok is not None).lower())
