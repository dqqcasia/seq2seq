# Implementation of the mainstream sequence to sequence models:

This project is originally forked from <https://github.com/chqiwang/transformer>. 

## Features
* Pre-processing script

## Usage
Create a new config file.

`cp config_template.yaml your_config.yaml`

Configure *train.src_path*, *train.dst_path*, *scr_vocab* and *dst_vocab* in *your_config.yaml*.
After that, run the following command to build the vocabulary files.

`python vocab.py -c your_config.yaml`
 
Edit *src\_vocab_size* and *dst\_vocab_size* in *your_config.yaml* according to the vocabulary files generated in previous step.

Run the following command to start training loops:

`python train.py -c your_config.yaml`

