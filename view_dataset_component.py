import streamlit as st
import pandas as pd

from settings import datasets_dir


def view_dataset_component():
    # state
    datasets = ["Please select a dataset"] + [f.name for f in datasets_dir.iterdir() if f.is_file() and f.name.endswith(".csv")]

    # methods

    # rendering
    st.write("## View any dataset")
    st.write("You can view any dataset here! Use this feature to validate that the preprocessing is correct.")

    st.write("####  Viewer settings:")
    selected_dataset = st.selectbox("Choose dataset", datasets)
    num_samples = st.number_input("Number of samples", min_value=5, max_value=100, value=10)
    if selected_dataset != "Please select a dataset":
        st.write("---")
        with st.spinner("Loading dataset..."):
            dataset_path = datasets_dir / selected_dataset
            df = pd.read_csv(dataset_path)
            st.write(f'Dataset {selected_dataset} has {len(df)} samples.')
            st.write(df.head(int(num_samples)))
