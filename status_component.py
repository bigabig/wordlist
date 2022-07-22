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


def render_status(status_message: StreamlitStatusMessage):
    match status_message.status:
        case StreamlitStatus.ERROR:
            st.error(status_message.message)
        case StreamlitStatus.WARNING:
            st.warning(status_message.message)
        case StreamlitStatus.INFO:
            st.info(status_message.message)
        case StreamlitStatus.SUCCESS:
            st.success(status_message.message)
        case _:
            st.error("Unknown status")


def status_component(status_message_key):
    if status_message_key in st.session_state and st.session_state[status_message_key]:
        if isinstance(st.session_state[status_message_key], list):
            for status_message in st.session_state[status_message_key]:
                render_status(status_message)
            st.session_state[status_message_key] = []

        else:
            render_status(st.session_state[status_message_key])
            st.session_state[status_message_key] = None
