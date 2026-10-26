"""Vercel Python entrypoint.

The Streamlit UI lives in streamlit_app.py. This module only exports the
ASGI app Vercel requires (`app`, `application`, or `handler`).
"""

import streamlit as st

app = st.App("streamlit_app.py")
