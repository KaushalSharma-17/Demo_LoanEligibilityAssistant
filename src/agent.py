"""
LangChain Agent combining all three tools:
- loan_policy_qa (RAG)
- eligibility_checker
- recommendation_engine
Uses Claude Haiku with ConversationBufferWindowMemory.
"""

import logging
import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic

from src.eligibility_checker import check_eligibility
from src.rag_pipeline import build_rag_chain
from src.recommendation_engine import generate_recommendations_from_json

load_dotenv()
logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are a professional and helpful AI Loan Assistant for a bank.
You help customers understand loan products, check their eligibility, and improve their financial profile.

You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Guidelines:
- For questions about loan policies, interest rates, documents, or products → use loan_policy_qa.
- For checking if someone is eligible → use eligibility_checker with their details as JSON.
- For improvement tips after rejection → use recommendation_engine.
- Always be polite, empathetic, and professional.
- Remember previous context from the conversation when answering follow-up questions.
- Use ₹ symbol for Indian Rupee amounts.

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


def build_agent(force_rebuild_rag: bool = False) -> AgentExecutor:
    """
    Build and return a LangChain AgentExecutor with all three tools and memory.

    Args:
        force_rebuild_rag: If True, rebuild the FAISS vector index from scratch.

    Returns:
        Configured AgentExecutor instance.
    """
    rag_chain = build_rag_chain(force_rebuild=force_rebuild_rag)

    def policy_qa_tool(query: str) -> str:
        """Answer questions from bank policy documents."""
        try:
            result = rag_chain.invoke({"query": query})
            return result.get("result", "No answer found in policy documents.")
        except Exception as e:
            logger.error(f"RAG tool error: {e}")
            return f"Error querying policy documents: {e}"

    tools = [
        Tool(
            name="loan_policy_qa",
            func=policy_qa_tool,
            description=(
                "Use this tool to answer questions about bank loan products, policies, "
                "interest rates, required documents, credit score guidelines, and eligibility criteria. "
                "Input should be a natural language question."
            ),
        ),
        Tool(
            name="eligibility_checker",
            func=check_eligibility,
            description=(
                "Use this tool to check if a customer is eligible for a loan. "
                "Input MUST be a valid JSON string with these fields: "
                "name (string), age (integer), salary (monthly in ₹, integer), "
                "loan_amount (integer), credit_score (integer). "
                'Example: {"name": "John", "age": 30, "salary": 60000, "loan_amount": 500000, "credit_score": 750}'
            ),
        ),
        Tool(
            name="recommendation_engine",
            func=generate_recommendations_from_json,
            description=(
                "Use this tool to generate personalized recommendations for an applicant who was NOT ELIGIBLE. "
                "Input must be a JSON string with two keys: "
                "'applicant' (dict with name, age, salary, loan_amount, credit_score) and "
                "'failed_rules' (list of rule dicts with keys: rule, message, passed, value). "
                "Use this after running eligibility_checker when the result is NOT ELIGIBLE."
            ),
        ),
    ]

    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        temperature=0,
    )

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=False,
    )

    prompt = PromptTemplate(
        template=AGENT_SYSTEM_PROMPT,
        input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )

    logger.info("LangChain Agent built successfully.")
    return agent_executor
