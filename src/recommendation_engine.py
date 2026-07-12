"""
Personalized Recommendation Engine.
Generates actionable improvement steps for applicants who are not eligible.
"""

import logging
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

load_dotenv()
logger = logging.getLogger(__name__)

RECOMMENDATION_PROMPT_TEMPLATE = """You are a helpful and empathetic bank financial advisor.
A loan applicant was found NOT ELIGIBLE based on the following assessment.

Applicant Profile:
{applicant_profile}

Failed Criteria:
{failed_criteria}

Your task:
1. Provide a numbered list of specific, actionable recommendations to improve eligibility.
2. For each recommendation, include a realistic timeframe.
3. If the loan amount is too high, suggest an alternative loan amount they may qualify for.
4. Suggest alternative loan products if applicable (e.g., secured loan, smaller personal loan).
5. End with an encouraging closing statement.

Keep the tone warm, professional, and motivating. Be specific — avoid generic advice.

Recommendations:"""


def generate_recommendations(applicant: dict, failed_rules: list[dict]) -> str:
    """
    Generate personalized improvement recommendations for an ineligible applicant.

    Args:
        applicant: Dictionary with applicant details (name, age, salary, loan_amount, credit_score).
        failed_rules: List of rule dicts that were not satisfied (from eligibility_checker).

    Returns:
        Formatted recommendation string.
    """
    if not failed_rules:
        return "The applicant meets all eligibility criteria. No recommendations needed."

    applicant_profile = (
        f"Name: {applicant.get('name')}\n"
        f"Age: {applicant.get('age')} years\n"
        f"Monthly Salary: ₹{applicant.get('salary', 0):,}\n"
        f"Loan Amount Requested: ₹{applicant.get('loan_amount', 0):,}\n"
        f"Credit Score: {applicant.get('credit_score')}"
    )

    failed_summary = "\n".join(
        [f"- {r['rule']}: {r['message']}" for r in failed_rules if not r["passed"]]
    )

    try:
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            temperature=0.3,
        )

        prompt = PromptTemplate(
            template=RECOMMENDATION_PROMPT_TEMPLATE,
            input_variables=["applicant_profile", "failed_criteria"],
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(
            applicant_profile=applicant_profile,
            failed_criteria=failed_summary,
        )

        logger.info(f"Recommendations generated for {applicant.get('name')}.")
        return response

    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        return f"Could not generate recommendations at this time. Please try again.\nError: {e}"


def generate_recommendations_from_json(input_json: str) -> str:
    """
    LangChain Tool-compatible wrapper for the recommendation engine.

    Args:
        input_json: JSON string with keys: applicant (dict) and failed_rules (list).

    Returns:
        Formatted recommendation string.
    """
    import json

    try:
        data = json.loads(input_json)
        applicant = data.get("applicant", {})
        failed_rules = data.get("failed_rules", [])
        return generate_recommendations(applicant, failed_rules)
    except Exception as e:
        return f"Invalid input to recommendation engine: {e}"
