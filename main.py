import streamlit as st
import gensim.downloader
import pandas as pd
from io import StringIO
import os
from pathlib import Path
import spacy


@st.experimental_singleton
def load_spacy():
    return spacy.load("en_core_web_sm")


@st.experimental_singleton
def load_model(modelname: str):
    print(modelname)
    return gensim.downloader.load(modelname)


glove_vectors = load_model('glove-twitter-25')
nlp = load_spacy()


def main():
    # State init
    if 'words' not in st.session_state:
        st.session_state.words = []
    if 'preds' not in st.session_state:
        st.session_state.preds = []
    if 'excludes' not in st.session_state:
        st.session_state.excludes = []
    if 'selected_preds' not in st.session_state:
        st.session_state.selected_preds = []
    if 'selected_ex' not in st.session_state:
        st.session_state.selected_preds = []
    if 'selected_preds' not in st.session_state:
        st.session_state.selected_preds = []

    def predict():
        st.session_state.preds = []
        if len(st.session_state.words) > 0 or len(st.session_state.excludes) > 0:
            st.session_state.preds = [x[0] for x in glove_vectors.most_similar(positive=st.session_state.words,
                                                                               negative=st.session_state.excludes,
                                                                               topn=num_predictions)]

    def model_change_callback():
        global glove_vectors
        glove_vectors = load_model(st.session_state.model_name)
        st.session_state.words = []
        st.session_state.preds = []
        st.session_state.excludes = []

    st.write("""
    # Wordlist Tool
    """)

    with st.sidebar:

        st.write("""
        ## Settings
        """)

        num_predictions = st.slider("# predictions", 1, 25, 10)

        st.selectbox("Which model do you want to use?", ['glove-twitter-25', 'glove-wiki-gigaword-100'], key='model_name',
                     on_change=model_change_callback)

    st.write("## Create a wordlist")

    def remove_words_callback():
        st.session_state.words = [word for i, word in enumerate(st.session_state.words) if not st.session_state.selected_words[i]]
        predict()

    def remove_excludes_callback():
        st.session_state.excludes = [word for i, word in enumerate(st.session_state.excludes) if not st.session_state.selected_excludes[i]]
        print(st.session_state.excludes)
        predict()

    def add_words_callback():
        added_preds = [word for i, word in enumerate(st.session_state.preds) if st.session_state.selected_preds[i]]
        st.session_state.words.extend(added_preds)
        predict()

    def add_excludes_callback():
        added_excludes = [word for i, word in enumerate(st.session_state.preds) if st.session_state.selected_preds[i]]
        st.session_state.excludes.extend(added_excludes)
        predict()

    def form_callback():
        new_word = st.session_state.word_input.strip()
        if len(new_word) > 0 and new_word not in st.session_state.words:
            st.session_state.words.append(st.session_state.word_input)
            predict()


    def import_callback():
        file = st.session_state.import_file
        if file:
            df = pd.read_csv(file, index_col="id")
            st.session_state.words = list(df['word'])
            predict()

    with st.form(key='my_form', clear_on_submit=True):
        st.text_input(label="Word input", key='word_input', placeholder="Please type a word")
        st.form_submit_button(label='Add word to wordlist', on_click=form_callback)

    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    col1, col2, col3 = st.columns(3)


    with col1:
        st.write("### Wordlist:")
        st.session_state.selected_words = [st.checkbox(word, key=f"words-{word}") for word in st.session_state.words]

    with col2:
        st.write("### Predictions:")
        st.session_state.selected_preds = [st.checkbox(word, key=f"predictions-{word}") for word in st.session_state.preds]

    with col3:
        st.write("### Exclude:")
        st.session_state.selected_excludes = [st.checkbox(word, key=f"excludes-{word}") for word in st.session_state.excludes]

    with c1:
        st.button("Remove", on_click=remove_words_callback, key="remove-words", disabled=sum(st.session_state.selected_words) == 0)

    with c2:
        st.button("<- Add", on_click=add_words_callback, disabled=sum(st.session_state.selected_preds) == 0)

    with c3:
        st.button("Add ->", on_click=add_excludes_callback, disabled=sum(st.session_state.selected_preds) == 0)

    with c4:
        st.button("Remove", on_click=remove_excludes_callback, key="remove-exludes", disabled=sum(st.session_state.selected_excludes) == 0)

    # st.button("Add", on_click=add_word_callback, kwargs={"word": word}, key=f'add-button-{word}')

    st.write("---")

    export_col, import_col = st.columns(2)

    with export_col:
        st.write("## Export")
        st.write("Press the button below to export the above wordlist as csv file.")
        st.download_button(label="Download wordlist as CSV",
                           data=pd.DataFrame([st.session_state.words], index=["word"]).T.to_csv(index_label="id"),
                           file_name='wordlist.csv',
                           mime='text/csv',
                           disabled=len(st.session_state.words) == 0)

    with import_col:
        st.write("## Import")
        st.write("Attention: Importing a file deletes the current wordlist!")
        st.file_uploader("Upload wordlist csv file", type=["csv"], accept_multiple_files=False, on_change=import_callback, key="import_file", help="Test")

    st.write("---")

    st.write("## Train new model")

    st.write("You can train your own model! Please upload training data, review the training data and start the model training :)")

    st.write("### Step 1: Upload your dataset")

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
                st.error("Dataset already exists!")

    st.button("Save dataset!", on_click=upload_dataset_callback, disabled=len(st.session_state.uploaded_files) == 0)

    st.write("### Step 2: Train model")

    st.text_input("Model name", key="new_model_name")

    datasets_dir = Path() / "datasets"
    datasets = [f.name for f in datasets_dir.iterdir() if f.is_dir()]
    st.selectbox("Choose dataset", datasets, key="selected_dataset")

    def train_callback():
        new_model_name = st.session_state.new_model_name.strip()
        selected_dataset = st.session_state.selected_dataset
        if len(new_model_name) > 0 and selected_dataset:
            dataset_dir = datasets_dir / selected_dataset

            print("Training!")

    st.button("Start training!", on_click=train_callback)


if __name__ == '__main__':
    main()
