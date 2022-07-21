import time
from io import StringIO
import os
from pathlib import Path
import streamlit as st
import pandas as pd
from spacy import Language
from gensim.models.word2vec import PathLineSentences
from gensim.models import Word2Vec
import numpy as np

from settings import datasets_dir, model_dir


def training_callback():
    model_name = st.session_state.new_model_name.strip()
    dataset_name = st.session_state.selected_dataset

    dataset_path = datasets_dir / dataset_name
    model_path = model_dir / model_name

    if model_path.exists():
        st.session_state.train_error_message = f"Model with name {model_name} already exists!"
        return

    # start training
    st.session_state.is_training = True
    time.sleep(10)

    # # load dataset
    # dataset = PathLineSentences(str(dataset_path))
    #
    # # train model
    # model = Word2Vec(sentences=dataset, vector_size=100, window=5, min_count=1, workers=4)
    #
    # # save model
    # model.wv.save(str(model_path))

    # end training
    st.session_state.is_training = False


def train_component():
    # State init
    if 'train_error_message' not in st.session_state:
        st.session_state.train_error_message = None
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False

    # methods

    # rendering
    st.write("### Step 2: Train model")

    if st.session_state.train_error_message:
        st.error(st.session_state.train_error_message)
        st.session_state.train_error_message = None

    st.text_input("Model name", key="new_model_name", disabled=st.session_state.is_training)

    datasets = [f.name for f in datasets_dir.iterdir() if f.is_dir()]
    st.selectbox("Choose dataset", datasets, key="selected_dataset", disabled=st.session_state.is_training)

    st.button("Start training!", on_click=training_callback, disabled=len(st.session_state.new_model_name.strip()) == 0 or not st.session_state.selected_dataset or st.session_state.is_training)

    if st.session_state.is_training:
        st.spinner('Model is training. Please wait...')

    # df = pd.DataFrame(
    #     np.zeros((1000, 2)),
    #     columns=["filename", "content"])
    # st.dataframe(df)
