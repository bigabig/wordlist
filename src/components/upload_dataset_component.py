import os
import threading
import time
from io import StringIO
from typing import List
import pandas as pd
import streamlit as st
from streamlit.uploaded_file_manager import UploadedFile
from config import datasets_dir
from components.status_component import status_component, StreamlitStatusMessage, StreamlitStatus

prepro_threads = []
prepro_thread_state = {}


def thread_watcher():
    while True:
        for thread in prepro_threads:
            if not thread.is_alive():
                del prepro_thread_state[thread.ident]
                prepro_threads.remove(thread)
        time.sleep(1)


prepro_thread_watcher = threading.Thread(target=thread_watcher, daemon=True)
prepro_thread_watcher.start()


def preprocess_text(text: str, lower_casing: bool, stop_word_filtering: bool, stop_words: List[str], pos_tag_filtering: bool, pos_tag_filter_list: List[str]) -> List[List[str]]:
    from main import nlp

    doc = nlp(text)

    sentences = []
    for sent in doc.sents:
        tokens = []
        for token in sent:
            if pos_tag_filtering and token.pos_ in pos_tag_filter_list:
                continue
            if stop_word_filtering:
                if len(stop_words) > 0:
                    if token.text in stop_words:
                        continue
                elif token.is_stop:
                    continue
            tokens.append(token.text.lower() if lower_casing else token.text)
        sentences.append(tokens)

    return sentences


def preprocessing_thread(files: UploadedFile | List[UploadedFile], dataset_path, dataset_file_path, lower_casing, stop_word_filtering, stop_word_list, pos_tag_filtering, pos_tag_filter_list):
    # create dataset directory
    dataset_path.mkdir(parents=True)

    # preprocess
    data = []

    # if we are provided with a list, it is a list of txt files
    if isinstance(files, list):
        for (i, file) in enumerate(files):
            prepro_thread_state[threading.get_ident()] = f'{i} / {len(files)}'
            sentences = preprocess_text(StringIO(file.getvalue().decode("utf-8")).read(), lower_casing, stop_word_filtering, stop_word_list, pos_tag_filtering, pos_tag_filter_list)
            data.append((file.name, "\n".join([" ".join(sent) for sent in sentences])))

    else:
        # read csv file
        df = pd.read_csv(files)

        # preprocess content
        for (idx, content) in df["article"].iteritems():
            prepro_thread_state[threading.get_ident()] = f"{idx} / {len(df['article'])}"
            sentences = preprocess_text(content, lower_casing, stop_word_filtering, pos_tag_filtering, pos_tag_filter_list)
            data.append((f"{files.name}-{idx}.txt", "\n".join([" ".join(sent) for sent in sentences])))

    # write dataset to directory
    for (filename, content) in data:
        with open(os.path.join(dataset_path.absolute(), filename), "w") as f:
            f.write(content)

    # write dataset to csv file
    pd.DataFrame(data, columns=["filename", "article"]).to_csv(dataset_file_path, index=False)


def upload_dataset_component():
    # state
    if 'dataset_status' not in st.session_state:
        st.session_state.dataset_status = []

    # methods
    def upload_dataset_callback():
        # validate input
        dataset_name = st.session_state.new_dataset_name.strip()

        if not st.session_state.uploaded_files or (isinstance(st.session_state.uploaded_files, list) and len(st.session_state.uploaded_files) == 0):
            st.session_state.dataset_status.append(StreamlitStatusMessage(
                status=StreamlitStatus.ERROR,
                message="Please provide training data!"))

        if len(dataset_name) == 0:
            st.session_state.dataset_status.append(StreamlitStatusMessage(
                status=StreamlitStatus.ERROR,
                message="Please provide a dataset name!"))

        # validate that dataset does not exist
        dataset_path = datasets_dir / dataset_name
        dataset_file_path = datasets_dir / f'{dataset_name}.csv'

        if len(dataset_name) > 0 and (dataset_path.exists() or dataset_file_path.exists()):
            st.session_state.dataset_status.append(StreamlitStatusMessage(status=StreamlitStatus.ERROR,
                                                                          message=f"Dataset '{dataset_name}' already exists!"))

        # validate csv file
        if isinstance(st.session_state.uploaded_files, UploadedFile) and st.session_state.uploaded_files.type == "text/csv":

            # validate column name
            column_name = st.session_state.content_column.strip()
            if len(column_name) == 0:
                st.session_state.dataset_status.append(StreamlitStatusMessage(
                    status=StreamlitStatus.ERROR,
                    message="Please provide a column name!"))

            try:
                df = pd.read_csv(st.session_state.uploaded_files)

                # validate that column is present in csv file
                if len(column_name) > 0 and column_name not in df.columns:
                    st.session_state.dataset_status.append(StreamlitStatusMessage(
                        status=StreamlitStatus.ERROR,
                        message=f"Column '{column_name}' not found in csv file!"))

            except Exception as e:
                st.session_state.dataset_status.append(StreamlitStatusMessage(status=StreamlitStatus.ERROR,
                                                                              message=f"Invalid CSV file: {e}"))

        # todo develop custom list component
        # todo deploy on server

        # validate pos tag filter list
        if st.session_state.pos_tag_filtering and len(st.session_state.pos_tags_to_filter_out) == 0:
            st.session_state.dataset_status.append(StreamlitStatusMessage(
                status=StreamlitStatus.ERROR,
                message="Please provide at least one POS tag to filter out!"
            ))

        # validate stopword list
        stop_words = []
        if st.session_state.stop_word_filtering and st.session_state.stop_word_file is not None:
            try:
                stop_words = list(filter(lambda x: len(x) > 0, st.session_state.stop_word_file.getvalue().decode("utf-8").splitlines()))
                print(stop_words)
                if len(stop_words) == 0:
                    st.session_state.dataset_status.append(StreamlitStatusMessage(
                        status=StreamlitStatus.ERROR,
                        message="Uploaded stop word list contains no words!"
                    ))
            except Exception as e:
                st.session_state.dataset_status.append(StreamlitStatusMessage(
                    status=StreamlitStatus.ERROR,
                    message=f"Invalid stopword list: {e}"
                ))

        if len(st.session_state.dataset_status) > 0:
            return

        # everything is okay
        # start preprocessing thread
        x = threading.Thread(target=preprocessing_thread,
                             args=(st.session_state.uploaded_files,
                                   dataset_path,
                                   dataset_file_path,
                                   st.session_state.lower_casing,
                                   st.session_state.stop_word_filtering,
                                   stop_words,
                                   st.session_state.pos_tag_filtering,
                                   st.session_state.pos_tags_to_filter_out),
                             name=dataset_name)
        x.start()
        prepro_thread_state[x.ident] = 0.0
        prepro_threads.append(x)
        st.session_state.dataset_status.append(StreamlitStatusMessage(StreamlitStatus.SUCCESS, f"Successfully started processing of dataset '{dataset_name}'!"))

    # rendering
    st.write("## Upload your dataset")
    st.write("You can upload a dataset to the server. The dataset will be preprocessed and stored in the server. You can then use the dataset to train a new model.")

    if len(prepro_threads) > 0:
        message = ""
        for thread in prepro_threads:
            message += f"- Dataset '{thread.name}' is currently processed. Please wait... ({prepro_thread_state[thread.ident]})\n"
        st.markdown(message)

    status_component("dataset_status")

    st.write("#### Dataset settings:")
    st.text_input("Dataset name", key="new_dataset_name")
    file_type = st.radio(
        "Format of your dataset",
        ('.txt files', '.csv file'),
        horizontal=True)
    if file_type == '.txt files':
        st.file_uploader("Choose text files", type=["txt"], accept_multiple_files=True, key="uploaded_files")
    elif file_type == '.csv file':
        st.text_input("Column name of text content", value="article", key="content_column", help="This column name must be present in the csv file.")
        st.file_uploader("Choose csv file", type=["csv"], accept_multiple_files=False, key="uploaded_files")

    st.write("#### Pre-processing settings:")
    st.checkbox('To lower case', key="lower_casing", disabled=True, value=True)
    stop_word_filtering = st.checkbox('Stop word filtering', key="stop_word_filtering", value=True)
    if stop_word_filtering:
        st.write("You can provide your own list of stop words. If you leave this field empty, the default list will be used. The file should contain one stop word per line.")
        st.file_uploader("Choose stop word file", type=["txt"], accept_multiple_files=False, key="stop_word_file")
    pos_tag_filtering = st.checkbox('POS tag filtering', key="pos_tag_filtering", value=True)
    if pos_tag_filtering:
        st.multiselect(
            'Choose POS tags to filter out',
            ["PUNCT", "SPACE", "NUM", "ADJ", "ADV", "NOUN", "VERB", "PRON", "DET", "PROPN", "CONJ"],
            ["PUNCT", "SPACE", "NUM"],
            key="pos_tags_to_filter_out")
    st.button("Upload dataset & start pre-processing!", on_click=upload_dataset_callback)
