import streamlit as st
import openai

from virtual_mentor.utils import create_message, compute_message_cost, query_model


historical_figures = [
    "Albert Einstein",
    "Napoleon Bonaparte",
    "Barack Obama",
    "Donald Trump",
    "Elon Musk",
    "Marie Curie",
]

HF_KEY = st.secrets["HF_KEY"]
openai.api_key = st.secrets["OPENAI_KEY"]
API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
headers = {"Authorization": f"Bearer {HF_KEY}"}


st.set_page_config(
    page_title="Your virtual mentor",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)


st.title("Your virtual mentor")

with st.form(key='my_form'):
    historical_character = st.selectbox("Choose a historical figure", historical_figures)
    context_msg_content = f"You are {historical_character}. Represent yourself in a way that is\
        consistent with your historical reputation."

    question_msg_content = st.text_area(f'Enter your question for {historical_character}',
                          "What is your opinion on the current political situation?")
    messages=[
            create_message('system', context_msg_content),
            create_message('user', question_msg_content),
        ]

    submitted = st.form_submit_button('Submit the question!')
    if submitted:
        response = query_model(messages)
        message_cost = compute_message_cost(messages)
        st.write(f"Message cost: {message_cost * 100}¢")
        st.write(response['choices'][0]['message']['content'])
