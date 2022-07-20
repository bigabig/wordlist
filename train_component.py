import time
from io import StringIO
import os
from pathlib import Path
import streamlit as st
import pandas as pd
from spacy import Language
from gensim.models.word2vec import PathLineSentences
from gensim.models import Word2Vec

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

    # load dataset
    dataset = PathLineSentences(str(dataset_path))

    # train model
    model = Word2Vec(sentences=dataset, vector_size=100, window=5, min_count=1, workers=4)

    # save model
    model.wv.save(str(model_path))

    # end training
    st.session_state.is_training = False


def train_component(nlp: Language):
    # State init
    if 'dataset_error_message' not in st.session_state:
        st.session_state.dataset_error_message = None
    if 'train_error_message' not in st.session_state:
        st.session_state.train_error_message = None
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False

    st.write("## Train new model")
    st.write("You can train your own model! Please upload training data, review the training data and start the model training :)")

    st.write("### Step 1: Upload your dataset")

    if st.session_state.dataset_error_message:
        st.error(st.session_state.dataset_error_message)
        st.session_state.dataset_error_message = None

    st.text_input("Dataset name", key="new_dataset_name")
    st.file_uploader("Choose text files", type=["txt"], accept_multiple_files=True, key="uploaded_files")
    if len(st.session_state.uploaded_files) > 0:
        data = []
        for file in st.session_state.uploaded_files:
            doc = nlp(StringIO(file.getvalue().decode("utf-8")).read())
            sentences = [[token.text.lower() for token in sent if
                          token.pos_ not in ("PUNCT", "SPACE", "NUM") and token.text.lower() not in nlp.Defaults.stop_words]
                         for
                         sent in doc.sents]
            data.append([file.name, "\n".join([" ".join(sent) for sent in sentences])])

        df = pd.DataFrame(
            data,
            columns=["filename", "content"])
        st.write("Please review the training data:")
        st.dataframe(df)

    def upload_dataset_callback():
        new_dataset_name = st.session_state.new_dataset_name.strip()
        if len(new_dataset_name) > 0 and len(data) > 0:
            p = Path() / "datasets" / new_dataset_name
            print(p.absolute())
            if not p.exists():
                p.mkdir(parents=True)
                for x in data:
                    filename = x[0]
                    content = x[1]
                    # write file
                    with open(os.path.join(p.absolute(), filename), "w") as f:
                        f.write(content)
            else:
                st.session_state.dataset_error_message = f"Dataset with name {new_dataset_name} already exists!"
        else:
            st.session_state.dataset_error_message = "Please provide a dataset name and files!"

    st.button("Save dataset!", on_click=upload_dataset_callback, disabled=len(st.session_state.uploaded_files) == 0 or len(st.session_state.new_dataset_name.strip()) == 0)

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
