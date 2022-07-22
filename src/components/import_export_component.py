import streamlit as st
import pandas as pd


def import_export(predict):
    def import_callback():
        file = st.session_state.import_file
        if file:
            df = pd.read_csv(file, index_col="id")
            st.session_state.words = list(df['word'])
            predict()
        st.success("Successfully imported wordlist!")

    export_col, import_col = st.columns(2)

    with export_col:
        st.write("## Export")
        st.write("Press the button below to export your wordlist as csv file.")
        st.download_button(label="Download wordlist as CSV",
                           data=pd.DataFrame([st.session_state.words], index=["word"]).T.to_csv(index_label="id"),
                           file_name='wordlist.csv',
                           mime='text/csv',
                           disabled=len(st.session_state.words) == 0)

    with import_col:
        st.write("## Import")
        st.write("Attention: Importing a file deletes the current wordlist!")
        st.file_uploader("Upload wordlist csv file",
                         type=["csv"],
                         accept_multiple_files=False,
                         on_change=import_callback,
                         key="import_file",
                         help="Test")
