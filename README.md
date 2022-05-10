# amr-qa
This repo maps natural language questions to SPARQL queries via AMR representations

## Project setup
### Library installations
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data downloads and configuration
```
# download datasets and AMR models
python data/download.py \
    --save-dir <your-local-directory>

# install the BLINK entity linker
# https://amrlib.readthedocs.io/en/latest/wiki/ 
```

### Install the AMR-to-text aligner (fast align)
```
# https://github.com/clab/fast_align
sudo apt-get install libgoogle-perftools-dev libsparsehash-dev
git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
```

## Run experiment
### Generate SPARQL queries for QALD-9 questions
```
# all questions
python -m amrqa.sparql.sparql \
    --data-filepath ./amrqa/sparql/qald_9.json \
    --fast-align-dir ~/Documents/fast_align/build \
    --propbank-filepath ~/Documents/data/amr-qa/probbank-dbpedia.pkl \
    --save-dir ~/Documents/data/amr-qa/generate/v2

# select questions (prints SPARQL to the command-line)
python -m amrqa.sparql.sparql \
    --data-filepath ./amrqa/sparql/qald_9.json \
    --fast-align-dir ~/Documents/fast_align/build \
    --propbank-filepath ~/Documents/data/amr-qa/probbank-dbpedia.pkl \
    --index 254
```

### Query the SPARQL API with the generated SPARQL queries
```
python -m amrqa.query \
    --query-file ~/Documents/data/amr-qa/generate/v2/generated_queries.json \
    --save-file ~/Documents/data/amr-qa/generate/v2/generated_results.json
```

### Evaluate the system by comparing the answers from generated queries to answers from ground-truth queries
```
python -m amrqa.evaluate \
    --ground-truth ~/Documents/data/amr-qa/evaluate/qald9_result.json \
    --predictions ~/Documents/data/amr-qa/generate/v2/generated_results.json \
    --save-dir ~/Documents/data/amr-qa/evaluate/v2
```

### Run tests
```
# within the main directory
python -m pytest
```