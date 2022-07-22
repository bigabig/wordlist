import streamlit as st


def wordlist(predict, vocab):
    # State init
    if 'wordlist_error_message' not in st.session_state:
        st.session_state.wordlist_error_message = None

    st.write("## Create a wordlist")

    if st.session_state.wordlist_error_message:
        st.error(st.session_state.wordlist_error_message)
        st.session_state.wordlist_error_message = None

    def remove_words_callback():
        st.session_state.words = [word for i, word in enumerate(st.session_state.words) if
                                  not st.session_state.selected_words[i]]
        predict()

    def remove_excludes_callback():
        st.session_state.excludes = [word for i, word in enumerate(st.session_state.excludes) if
                                     not st.session_state.selected_excludes[i]]
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
        if len(new_word) == 0:
            return

        if new_word in st.session_state.words:
            st.session_state.wordlist_error_message = f"Word {new_word} is already in the wordlist!"
            return

        if new_word not in vocab:
            st.session_state.wordlist_error_message = f"Word '{new_word}' is not contained in the model's vocabulary. Sorry :("
            return

        st.session_state.words.append(st.session_state.word_input)
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
        st.session_state.selected_preds = [st.checkbox(word, key=f"predictions-{word}") for word in
                                           st.session_state.preds]

    with col3:
        st.write("### Exclude:")
        st.session_state.selected_excludes = [st.checkbox(word, key=f"excludes-{word}") for word in
                                              st.session_state.excludes]

    with c1:
        st.button("Remove", on_click=remove_words_callback, key="remove-words",
                  disabled=sum(st.session_state.selected_words) == 0)

    with c2:
        st.button("<- Add", on_click=add_words_callback, disabled=sum(st.session_state.selected_preds) == 0)

    with c3:
        st.button("Add ->", on_click=add_excludes_callback, disabled=sum(st.session_state.selected_preds) == 0)

    with c4:
        st.button("Remove", on_click=remove_excludes_callback, key="remove-exludes",
                  disabled=sum(st.session_state.selected_excludes) == 0)