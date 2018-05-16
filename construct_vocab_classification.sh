for dataset in "TREC" "stsa.binary" "stsa.fine" \
               "custrev" "mpqa" "rt-polarity" "subj"
do
    python construct_vocab_classification.py --dataset ${dataset}
done
