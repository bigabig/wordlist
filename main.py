import streamlit as st
import gensim.downloader


@st.experimental_singleton
def load_model(modelname: str):
    print(modelname)
    return gensim.downloader.load(modelname)


glove_vectors = load_model('glove-twitter-25')


def main():
    st.write("""
    # Wordlist!
    """)

    # Initialization
    if 'words' not in st.session_state:
        st.session_state.words = []
    if 'preds' not in st.session_state:
        st.session_state.preds = ["tim"]

    def model_change_callback():
        global glove_vectors
        glove_vectors = load_model(st.session_state.model_name)
        st.session_state.words = []
        st.session_state.preds = []

    def add_word_callback(word: str):
        st.session_state.words.append(word)
        predict()

    def form_callback():
        st.session_state.words.append(st.session_state.word_input)
        predict()

    def predict():
        st.session_state.preds = [x[0] for x in glove_vectors.most_similar(positive=st.session_state.words, negative=[], topn=10)]

    col1, col2 = st.columns(2)

    with col1:
        st.write("## List:")
        for word in st.session_state.words:
            st.write(word)

    with col2:
        st.write("## Predictions:")
        for word in st.session_state.preds:
            st.write(word)
            st.button("Add", on_click=add_word_callback, kwargs={"word": word}, key=f'add-button-{word}')

    with st.form(key='my_form'):
        word_input = st.text_input("Word input", key='word_input')
        submit_button = st.form_submit_button(label='Submit', on_click=form_callback)

    st.selectbox("Which model do you want to use?", ['glove-twitter-25', 'glove-wiki-gigaword-100'], key='model_name', on_change=model_change_callback)


if __name__ == '__main__':
    main()
