import threading
import time
from pathlib import Path
import streamlit as st
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from config import datasets_dir, model_dir
from components.status_component import StreamlitStatusMessage, StreamlitStatus, status_component

train_threads = []
train_thread_state = {}


def thread_watcher():
    while True:
        for thread in train_threads:
            if not thread.is_alive():
                del train_thread_state[thread.ident]
                train_threads.remove(thread)
        time.sleep(1)


train_thread_watcher = threading.Thread(target=thread_watcher, daemon=True)
train_thread_watcher.start()


def training_thread(model_path: Path, dataset_path: Path, vector_size: int, window_size: int, min_count: int):
    # create model file
    model_path.touch()

    # start training
    train_thread_state[threading.get_ident()] = True

    # load dataset
    dataset = PathLineSentences(str(dataset_path))

    # train model
    model = Word2Vec(sentences=dataset, vector_size=vector_size, window=window_size, min_count=min_count, workers=4)

    # save model
    model.wv.save(str(model_path))

    # end training
    train_thread_state[threading.get_ident()] = False


def train_component():
    # State init
    datasets = ["Please select a dataset"] + [f.name.removesuffix(".csv") for f in datasets_dir.iterdir() if f.is_file() and f.name.endswith(".csv")]
    if 'train_status' not in st.session_state:
        st.session_state.train_status = []

    # methods
    def train_callback():
        model_name = st.session_state.training_model_name.strip()
        model_path = model_dir / model_name

        # validate that a dataset is selected
        if st.session_state.training_dataset is None or st.session_state.training_dataset == "Please select a dataset":
            st.session_state.train_status.append(StreamlitStatusMessage(StreamlitStatus.ERROR, "Please select a dataset!"))

        # validate that a model name is provided
        if st.session_state.training_model_name is None or len(model_name) == 0:
            st.session_state.train_status.append(StreamlitStatusMessage(StreamlitStatus.ERROR, "Please provide a model name!"))

        # validate that model does not exist
        if len(model_name) > 0:
            if model_path.exists():
                st.session_state.train_status.append(StreamlitStatusMessage(StreamlitStatus.ERROR,
                                                                            f"Model with name '{model_name}' already exists!"))

        # start training thread if no error
        if len(st.session_state.train_status) == 0:
            x = threading.Thread(target=training_thread,
                                 args=(model_path,
                                       datasets_dir / st.session_state.training_dataset,
                                       round(st.session_state.vector_size if "vector_size" in st.session_state else 100),
                                       round(st.session_state.window_size if "window_size" in st.session_state else 5),
                                       round(st.session_state.min_count if "min_count" in st.session_state else 1)),
                                 name=model_name)
            x.start()
            train_thread_state[x.ident] = True
            train_threads.append(x)
            st.session_state.train_status.append(StreamlitStatusMessage(StreamlitStatus.SUCCESS, f"Successfully started training of model '{model_name}'!"))

    # rendering
    st.write("## Train your own model")
    st.write("You can train your own model! Please select a dataset for training, then start the training process!")

    if len(train_threads) > 0:
        message = ""
        for thread in train_threads:
            message += f"- Model '{thread.name}' is currently training. Please wait...\n"
        st.markdown(message)

    status_component("train_status")

    st.write("#### Training settings:")
    st.text_input("Model name", key="training_model_name")
    st.selectbox("Choose dataset", datasets, key="training_dataset")
    with st.expander("Model parameters (only for advanced users)"):
        st.number_input("Vector size", min_value=50, max_value=500, value=100, key="vector_size")
        st.number_input("Window size", min_value=1, max_value=10, value=5, key="window_size")
        st.number_input("Min count", min_value=1, max_value=10, value=1, key="min_count")
    st.button("Start training!", on_click=train_callback)
