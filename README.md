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
