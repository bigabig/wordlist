import threading
import time
import streamlit as st
from streamlit.scriptrunner import add_script_run_ctx

threads = []
thread_state = {}
thread_watcher_thread = None


def thread_function():
    print("start")
    for i in range(10):
        thread_state[threading.get_ident()] = i
        time.sleep(1)
    print("end")


def thread_watcher():
    while True:
        for thread in threads:
            if not thread.is_alive():
                del thread_state[thread.ident]
                threads.remove(thread)
        time.sleep(1)


def thread_component():
    global thread_watcher_thread

    def test_callback():
        # start a new thread that does heavy computing
        x = threading.Thread(target=thread_function)
        add_script_run_ctx(x)
        thread_state[x.ident] = 0
        x.start()
        threads.append(x)

    st.button("Add!", on_click=test_callback)

    if len(threads) > 0:
        st.write("## Monitoring")
        st.button("Refresh", key="refresh")
        for thread in threads:
            st.write(thread.ident)
            st.progress(thread_state[thread.ident] * 10)

    if not thread_watcher_thread:
        thread_watcher_thread = threading.Thread(target=thread_watcher, daemon=True)
        add_script_run_ctx(thread_watcher_thread)
        thread_watcher_thread.start()

