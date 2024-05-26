import streamlit as st
from streamlit_chat import message
from agent import generate_agent, handle_react_chat

# Handle new user input
agent = generate_agent("Giới thiệu cho tôi bạn là ai")

# Setting page title and header
st.set_page_config(page_title="SOPE", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>SOPE - Your smart Shopping Assistant 😬</h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        print("user_input",user_input)
        output = handle_react_chat(agent,user_input)
        print("output",output)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output["output"])

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))