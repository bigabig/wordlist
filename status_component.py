from enum import Enum
import streamlit as st


class StreamlitStatus(Enum):
    ERROR = 1
    WARNING = 2
    INFO = 3
    SUCCESS = 4


class StreamlitStatusMessage:
    def __init__(self, status: StreamlitStatus, message: str):
        self.status = status
        self.message = message


def status_component(status_message_key):
    if status_message_key in st.session_state and st.session_state[status_message_key]:
        match st.session_state[status_message_key].status:
            case StreamlitStatus.ERROR:
                st.error(st.session_state[status_message_key].message)
            case StreamlitStatus.WARNING:
                st.warning(st.session_state[status_message_key].message)
            case StreamlitStatus.INFO:
                st.info(st.session_state[status_message_key].message)
            case StreamlitStatus.SUCCESS:
                st.success(st.session_state[status_message_key].message)
            case _:
                st.error("Unknown status")
        st.session_state[status_message_key] = None
