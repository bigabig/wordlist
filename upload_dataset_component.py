import os
import threading
import time
from io import StringIO
from typing import List
import streamlit as st
import pandas as pd
from streamlit.scriptrunner import add_script_run_ctx
from streamlit.uploaded_file_manager import UploadedFile
from settings import datasets_dir
from status_component import status_component, StreamlitStatusMessage, StreamlitStatus

threads = []
thread_state = {}


def thread_watcher():
    while True:
        for thread in threads:
            if not thread.is_alive():
                del thread_state[thread.ident]
                threads.remove(thread)
        time.sleep(1)


thread_watcher_thread = threading.Thread(target=thread_watcher, daemon=True)
thread_watcher_thread.start()


def preprocess_text(text: str) -> List[List[str]]:
    from main import nlp

    doc = nlp(text)
    sentences = [[token.text.lower() for token in sent if
                  token.pos_ not in (
                      "PUNCT", "SPACE", "NUM") and token.text.lower() not in nlp.Defaults.stop_words]
                 for
                 sent in doc.sents]
    return sentences


def preprocessing_thread(files: List[UploadedFile], dataset_path, dataset_file_path):
    # create dataset directory
    dataset_path.mkdir(parents=True)

    # preprocess
    data = []
    for (i, file) in enumerate(files):
        match file.type:
            case 'text/plain':
                sentences = preprocess_text(StringIO(file.getvalue().decode("utf-8")).read())
                data.append((file.name, "\n".join([" ".join(sent) for sent in sentences])))
            case 'text/csv':
                # read csv file
                df = pd.read_csv(file)

                # preprocess content
                for (idx, content) in df["article"].iteritems():
                    thread_state[threading.get_ident()] = idx / len(df["article"])
                    print(f"Preprocessing {idx}/{len(df['article'])}")
                    sentences = preprocess_text(content)
                    data.append((f"{file.name}-{idx}.txt", "\n".join([" ".join(sent) for sent in sentences])))

    # write dataset to directory
    for (filename, content) in data:
        with open(os.path.join(dataset_path.absolute(), filename), "w") as f:
            f.write(content)

    # write dataset to csv file
    pd.DataFrame(data, columns=["filename", "article"]).to_csv(dataset_file_path, index=False)


def upload_dataset_component():
    # state
    if 'dataset_status' not in st.session_state:
        st.session_state.dataset_status = None

    # methods
    def upload_dataset_callback():
        # validate input
        dataset_name = st.session_state.new_dataset_name.strip()

        if len(st.session_state.uploaded_files) == 0 and len(dataset_name) == 0:
            st.session_state.dataset_status = StreamlitStatusMessage(
                status=StreamlitStatus.ERROR,
                message="Please upload a dataset and provide a dataset name!"
            )
            return

        if len(st.session_state.uploaded_files) == 0:
            st.session_state.dataset_status = StreamlitStatusMessage(
                status=StreamlitStatus.ERROR,
                message="Please provide training data!")
            return

        if len(dataset_name) == 0:
            st.session_state.dataset_status = StreamlitStatusMessage(
                status=StreamlitStatus.ERROR,
                message="Please provide a dataset name!")
            return

        # validate that dataset does not exist
        dataset_path = datasets_dir / dataset_name
        dataset_file_path = datasets_dir / f'{dataset_name}.csv'

        if dataset_path.exists() or dataset_file_path.exists():
            st.session_state.dataset_status = StreamlitStatusMessage(status=StreamlitStatus.ERROR,
                                                                     message=f"Dataset '{dataset_name}' already exists!")
            return

        # todo: validate that uploaded fiels are valid
        # # validate csv files and file type
        # for file in st.session_state.uploaded_files:
        #     match file.type:
        #         case 'text/plain':
        #             # todo: check that format is UTF-8
        #             pass
        #         case 'text/csv':
        #             # read csv file
        #             df = pd.read_csv(file)
        #             print(df.head())
        #
        #             # validate that csv file contains content
        #             if "article" not in df.columns:
        #                 st.session_state.dataset_status = StreamlitStatusMessage(
        #                     StreamlitStatus.ERROR, f"No 'article' column in csv file {file.name}!")
        #                 return
        #
        #         case _:
        #             st.session_state.dataset_status = StreamlitStatusMessage(StreamlitStatus.ERROR,
        #                                                                      "Unknown file type")
        # everything is okay
        # start preprocessing thread
        x = threading.Thread(target=preprocessing_thread,
                             args=(st.session_state.uploaded_files, dataset_path, dataset_file_path),
                             name=dataset_name)
        thread_state[x.ident] = 0.0
        x.start()
        threads.append(x)

    # rendering
    st.write("### Step 1: Upload your dataset")

    if len(threads) > 0:
        st.write("Monitoring preprocessing progress:")
        st.button("Refresh", key="refresh")
        for thread in threads:
            st.write(thread.name)
            st.write(thread_state[thread.ident])
            st.progress(thread_state[thread.ident])

    status_component("dataset_status")

    with st.form(key='upload_dataset_form', clear_on_submit=True):
        st.text_input("Dataset name", key="new_dataset_name")
        st.file_uploader("Choose text files", type=["txt", "csv"], accept_multiple_files=True, key="uploaded_files")
        st.form_submit_button("Save dataset!", on_click=upload_dataset_callback)
