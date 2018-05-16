import argparse
import collections
import io
import json
import sys

# python construct_vocab.py --data datasets/wikitext-103/wiki.train.tokens -t 50 -s datasets/wikitext-103/vocab.t50.json # NOQA

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', required=True)
parser.add_argument('--threshold', '-t', type=int, default=50)
parser.add_argument('--save', '-s', default='vocab.json')
args = parser.parse_args()

count = collections.defaultdict(int)
with io.open(args.data, encoding='utf-8') as f:
    for line in f:
        words = line.split() + ['<eos>']
        for word in words:
            count[word] += 1

vocab = {'<eos>': 0, '<unk>': 1}
for w, c in sorted(count.items(), key=lambda x: (-x[1], x[0])):
    if c < args.threshold:
        continue
    if w not in vocab:
        vocab[w] = len(vocab)

print('# of words: {}'.format(len(vocab)))

json.dump(dict(vocab), open(args.save, 'w'))
json.dump(dict(count), open(args.save + '.count', 'w'))
