#!/usr/bin/env bash

set -x

DATA_DIR=$1
TOKENIZER=$2

TOOLS_DIR=../tools

## paths to training and development datasets
src_ext=zh
trg_ext=en

train_prefix=train
dev_prefix=dev
test_prefix=test

# path to moses
MOSES=$TOOLS_DIR/mosesdecoder/scripts
# path to subword nmt
SUBWORD_NMT=$TOOLS_DIR/subword-nmt
# path to createVocab
CREATEVOCAB=$TOOLS_DIR/createVocab.py

mkdir -p $DATA_DIR/processed/

declare -a arr_src=($train_prefix.$src_ext $dev_prefix.$src_ext $test_prefix.$src_ext)
declare -a arr_trg=($train_prefix.$trg_ext $dev_prefix.$trg_ext $test_prefix.$trg_ext)


######################
# jieba segmentation

for file in "${arr_src[@]}"
do
    python -m jieba -d " " $DATA_DIR/$file > $DATA_DIR/processed/$file.seg
done

######################
# moses truecase
$MOSES/recaser/train-truecaser.perl  --model $DATA_DIR/truecase_model/truecase_model.en --corpus $DATA_DIR/$train_prefix.$trg_ext

for file in "${arr_trg[@]}"
do
    $MOSES/recaser/truecase.perl --model $DATA_DIR/truecase_model/truecase_model.en < $DATA_DIR/$file > $DATA_DIR/processed/$file.true
done

######################
# tokenizer
if [ $2 == "moses" ];then
# moses tokenizer
    for file in "${arr_trg[@]}"
    do
        $MOSES/tokenizer/tokenizer.perl -l en < $DATA_DIR/processed/$file.true > $DATA_DIR/processed/$file.true.tok
    done
else
# nltk tokenizer
    for file in "${arr_trg[@]}"
    do
        python $TOOLS_DIR/tokenizer_nltk.py $DATA_DIR/processed/$file.true $DATA_DIR/processed/$file.true.tok
    done
fi

######################
# subword segmentation
mkdir -p $DATA_DIR/bpe_model
bpe_operations=30000

cat $DATA_DIR/$train_prefix.$src_ext.seg | $SUBWORD_NMT/learn_bpe.py -s $bpe_operations > $DATA_DIR/bpe_model/$train_prefix.$src_ext.seg.bpe.$bpe_operations.model

for file in "${arr_src[@]}"
do
    $SUBWORD_NMT/apply_bpe.py -c $DATA_DIR/bpe_model/$train_prefix.$src_ext.seg.bpe.$bpe_operations.model < $DATA_DIR/processed/$file.seg > $DATA_DIR/processed/$file.seg.bpe.$bpe_operations
done

cat $DATA_DIR/$train_prefix.$trg_ext.true.tok | $SUBWORD_NMT/learn_bpe.py -s $bpe_operations > $DATA_DIR/bpe_model/$train_prefix.$trg_ext.true.tok.bpe.$bpe_operations.model

for file in "${arr_trg[@]}"
do
    $SUBWORD_NMT/apply_bpe.py -c $DATA_DIR/bpe_model/$train_prefix.$trg_ext.true.tok.bpe.$bpe_operations.model < $DATA_DIR/processed/$file.true.tok > $DATA_DIR/processed/$file.true.tok.bpe.$bpe_operations
done

#create vocabuary
python $CREATEVOCAB $DATA_DIR/processed/$train_prefix.$src_ext.seg.bpe.$bpe_operations $DATA_DIR/processed/$train_prefix.$src_ext.seg.bpe.$bpe_operations.vocab
python $CREATEVOCAB $DATA_DIR/processed/$train_prefix.$trg_ext.true.tok.bpe.$bpe_operations $DATA_DIR/processed/$train_prefix.$trg_ext.true.tok.bpe.$bpe_operations.vocab
