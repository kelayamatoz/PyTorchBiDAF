## 0. Requirements
#### General
- Python (developed on 3.5.2. Issues have been reported with Python 2!)
- unzip

#### Python Packages
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)

## 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

## 2. Training
```
python -m basic.cli
```
