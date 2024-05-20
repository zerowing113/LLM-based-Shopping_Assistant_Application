from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import (
    ConversationSummaryBufferMemory,
    ConversationBufferWindowMemory,
)
from model import langchain_llm, llamaindex_embed_model, llamaindex_llm
from langchain_core.prompts import PromptTemplate
from retrieval import Retrieval

# Construct the JSON agent
from prompt import classify_prompt, react_prompt

classify_prompt = PromptTemplate.from_template(classify_prompt)
chain = LLMChain(llm=langchain_llm, prompt=classify_prompt)


def tool_get_RAG():
    """Returns the RAG"""
    from llama_index.core.tools import QueryEngineTool

    retrieval = Retrieval()
    index = retrieval.load_index(llamaindex_embed_model)
    query_engine = retrieval.create_query_engine(index, llamaindex_llm)
    tool = QueryEngineTool.from_defaults(
        query_engine,
        name="VectorDB",
        description="Đây là công cụ tìm kiếm sản phẩm của cửa hàng cellphones",
    )
    # run tool as langchain structured tool
    lc_tool = tool.as_langchain_tool()
    return lc_tool


def get_prompt_input(user_input: str):
    # Construct the JSON agent
    chain = LLMChain(llm=langchain_llm, prompt=classify_prompt)
    return chain.invoke(
        {
            "question": f"{user_input}",
            "context": """a. Đề xuất sản phẩm: Sử dụng lời nhắc này khi mục đích của người dùng là khám phá các sản phẩm mới hoặc tìm đề xuất dựa trên sở thích hoặc giao dịch mua trước đây của họ.
         b. Truy xuất thông tin sản phẩm: Sử dụng lời nhắc này khi người dùng tìm kiếm thông tin chi tiết về một sản phẩm cụ thể, chẳng hạn như tính năng, thông số kỹ thuật hoặc đánh giá.
         c. Câu hỏi thường gặp Trả lời: Sử dụng lời nhắc này khi truy vấn của người dùng phù hợp với các câu hỏi thường gặp hoặc giải quyết các vấn đề hỗ trợ kỹ thuật, chính sách của các sản phẩm.
         d. Thanh toán: Tận dụng lời nhắc này để tạo điều kiện thuận lợi cho quá trình mua hàng, bao gồm xử lý thông tin thanh toán, xác nhận đơn hàng và chi tiết giao hàng.
         e. Khác: Sử dụng lời nhắc này cho các truy vấn không phù hợp với các danh mục trước đó, chẳng hạn như cung cấp hỗ trợ chung, cung cấp hỗ trợ quản lý tài khoản hoặc xử lý phản hồi.""",
        }
    )


def handle_user_prompt(user_prompt):
    user_prompt = user_prompt.splitlines()
    return [
        line.split(": ", maxsplit=1)[1]
        for line in user_prompt
        if line.strip().startswith("Answer:") or line.strip().startswith("Explain:")
    ]


def handle_conversation_turn(user_input: str):
    while True:
        user_prompt = get_prompt_input(user_input)
        return_prompt = handle_user_prompt(user_prompt["text"])
        if return_prompt is not None:
            try:
                if (
                    "Đề xuất sản phẩm" in return_prompt[0]
                    or "Truy xuất thông tin sản phẩm" in return_prompt[0]
                    or "Câu hỏi thường gặp Trả lời" in return_prompt[0]
                    or "Thanh toán" in return_prompt[0]
                    or "Khác" in return_prompt[0]
                ):
                    if len(return_prompt) > 1:
                        return return_prompt[-1]
                    return return_prompt[0]
            except:
                print("Error")


tools = [tool_get_RAG()]


def generate_agent(input):
    user_prompt = handle_conversation_turn(input)
    print(user_prompt)
    react_prompt = PromptTemplate.from_template(
        """
                Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful tool that can {user_prompt}. 


        TOOLS:
        ------

        Assistant has access to the following tools:

        {tools}

        To use a tool, please use the following format:

        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat TWICE!!!)
        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```
        -Please ensure that the answers are as emotionally rich and detailed as possible. 
        - Final Answer should respond in TRADITIONAL VIETNAMESE but Chain of thought steps and action steps are in English
        - If there is a Final Answer, return the result
        - Please try to use a three part structure to output the answer, and try to segment it according to the key points. The answer should be no less than 300 words!!!
        Let's begin!

        New input: {input}
        {agent_scratchpad}
        """,
        partial_variables={"user_prompt": user_prompt},
    )
    # react_prompt.format(user_prompt=user_prompt)

    # Construct the ReAct agent

    # Previous conversation history:
    #     {chat_history}
    agent = create_react_agent(langchain_llm, tools, react_prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        # memory=ConversationBufferWindowMemory(
        #     k=5, memory_key="chat_history", return_messages=True
        # ),
        handle_parsing_errors=True,
    )
    return agent_executor


def load_agent(input):
    agent = generate_agent(input)
    return agent


def handle_react_chat(agent, input):
    # print(agent.memory.load_memory_variables({}))
    return agent.invoke({"input": input})


if __name__ == "__main__":
    agent = load_agent("Giới thiệu cho tôi những mẫu điện thoại dưới 700k đáng xem")
    print(handle_react_chat(agent, "Giới thiệu cho tôi những mẫu điện thoại dưới 700k đáng xem"))
