import os
from pathlib import Path

import streamlit as st

from app.main import build_default_services


st.set_page_config(page_title='Tumor Agent RAG', page_icon='🧠', layout='wide')
st.title('Tumor Agent RAG + Multimodal Medical AI')
st.caption('CrewAI agents, LangGraph workflow, LangChain retrieval, and medical evidence routing.')

services = build_default_services()

st.sidebar.header('Services')
st.sidebar.write('Classification:', services['classification'].__class__.__name__)
st.sidebar.write('Detection:', services['detection'].__class__.__name__)
st.sidebar.write('Segmentation:', services['segmentation'].__class__.__name__)
st.sidebar.write('STT:', services['stt'].__class__.__name__)
st.sidebar.write('Retriever:', services['retriever'].__class__.__name__)
st.sidebar.write('Web:', services['web'].__class__.__name__)
st.sidebar.write('Crew:', services['crew'].__class__.__name__)
st.sidebar.write('Graph:', services['graph'].__class__.__name__)

st.header('Medical Query')
user_query = st.text_area('Enter a clinical question or evidence request:', value='Summarize the current tumor image evidence.')
image_path = st.text_input('Optional image path:', value='')
audio_path = st.text_input('Optional audio path:', value='')

if st.button('Run workflow'):
    state = {
        'session_id': 'streamlit-session',
        'user_query': user_query,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'document_paths': [],
        'retrieved_evidence': [],
        'web_evidence': [],
        'agent_findings': {},
        'final_response': None,
        'errors': [],
    }

    try:
        serviced_state = services['graph'].run(state)
        services['crew'].run(user_query)
        st.success('Workflow completed')
        st.json(serviced_state)
    except Exception as exc:
        st.error(str(exc))
