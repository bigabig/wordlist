import streamlit as st
import gensim.downloader
import spacy
from import_export import import_export
from settings import model_dir
from train_component import train_component
from wordlist import wordlist
from gensim.models import KeyedVectors

available_gensim_models = ['glove-twitter-25', 'glove-wiki-gigaword-100']
available_user_models = [f.name for f in model_dir.iterdir() if f.is_file()]


@st.experimental_singleton
def load_spacy():
    print("en_core_web_sm")
    return spacy.load("en_core_web_sm")


@st.experimental_singleton
def load_model(modelname: str):
    if modelname in available_gensim_models:
        return gensim.downloader.load(modelname)
    return KeyedVectors.load(str((model_dir / modelname).absolute()), mmap='r')


glove_vectors = None
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
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "glove-twitter-25"

    glove_vectors = load_model(st.session_state.model_name)
    vocab = glove_vectors.key_to_index.keys()

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

        st.selectbox("Which model do you want to use?", available_gensim_models + available_user_models, key='model_name',
                     on_change=model_change_callback)

    tab1, tab2, tab3 = st.tabs(["Create a wordlist", "Import & Export", "Train a model"])

    with tab1:
        wordlist(predict, vocab)

    with tab2:
        import_export(predict)

    with tab3:
        train_component(nlp)


if __name__ == '__main__':
    main()
