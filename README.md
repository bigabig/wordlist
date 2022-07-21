# wordlist

### start
```
streamlit start main.py
```

### conda environment
```
conda create -n wordlist python=3.10
conda activate wordlist
conda install streamlit
conda install -c conda-forge spacy
conda install -c conda-forge cupy
python -m spacy download en_core_web_sm
pip install gensim
```
