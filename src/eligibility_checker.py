"""
Loan Eligibility Checker.
Applies hard business rules and uses Claude Haiku to format the final response.
"""

import json
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

load_dotenv()
logger = logging.getLogger(__name__)

ELIGIBILITY_PROMPT_TEMPLATE = """You are a professional bank loan officer. Based on the eligibility 
check results below, write a clear and empathetic response for the applicant.

Applicant Profile:
{applicant_profile}

Rule Check Results:
{rule_results}

Overall Verdict: {verdict}

Instructions:
- State the verdict clearly at the top (ELIGIBLE or NOT ELIGIBLE).
- List each rule with a ✅ (pass) or ❌ (fail) indicator.
- For failed rules, explain what the applicant needs to improve.
- Keep the tone professional and respectful.
- End with a brief next-steps sentence.

Response:"""


@dataclass
class EligibilityResult:
    """Holds the structured result of an eligibility check."""
    eligible: bool
    verdict: str
    rule_results: list[dict]
    applicant: dict
    formatted_response: str


def apply_business_rules(applicant: dict) -> tuple[bool, list[dict]]:
    """
    Apply hard eligibility rules and return overall result and per-rule breakdown.

    Rules:
    - Age >= 21
    - Monthly salary >= 30,000
    - Credit score >= 700
    - Loan amount <= 10 × monthly salary
    """
    rules = []
    all_passed = True

    age = applicant.get("age", 0)
    age_pass = age >= 21
    rules.append({
        "rule": "Age >= 21 years",
        "value": age,
        "passed": age_pass,
        "message": f"Age is {age} years — {'satisfies' if age_pass else 'does not satisfy'} the minimum age requirement of 21 years.",
    })
    if not age_pass:
        all_passed = False

    salary = applicant.get("salary", 0)
    salary_pass = salary >= 30000
    rules.append({
        "rule": "Monthly Salary >= ₹30,000",
        "value": salary,
        "passed": salary_pass,
        "message": f"Monthly salary is ₹{salary:,} — {'meets' if salary_pass else 'does not meet'} the minimum requirement of ₹30,000.",
    })
    if not salary_pass:
        all_passed = False

    credit_score = applicant.get("credit_score", 0)
    credit_pass = credit_score >= 700
    rules.append({
        "rule": "Credit Score >= 700",
        "value": credit_score,
        "passed": credit_pass,
        "message": f"Credit score is {credit_score} — {'acceptable' if credit_pass else 'below the minimum acceptable score of 700'}.",
    })
    if not credit_pass:
        all_passed = False

    loan_amount = applicant.get("loan_amount", 0)
    max_eligible = salary * 10
    loan_pass = loan_amount <= max_eligible
    rules.append({
        "rule": f"Loan Amount <= 10 × Monthly Salary (max ₹{max_eligible:,})",
        "value": loan_amount,
        "passed": loan_pass,
        "message": f"Requested loan of ₹{loan_amount:,} is {'within' if loan_pass else 'above'} the maximum eligible amount of ₹{max_eligible:,}.",
    })
    if not loan_pass:
        all_passed = False

    return all_passed, rules


def check_eligibility(input_json: str) -> str:
    """
    Main eligibility checker function exposed as a LangChain Tool.

    Args:
        input_json: JSON string with keys: name, age, salary, loan_amount, credit_score.

    Returns:
        Formatted natural-language eligibility response string.
    """
    try:
        applicant = json.loads(input_json)
    except json.JSONDecodeError as e:
        return f"Invalid input format. Please provide a valid JSON string. Error: {e}"

    required_fields = ["name", "age", "salary", "loan_amount", "credit_score"]
    missing = [f for f in required_fields if f not in applicant]
    if missing:
        return f"Missing required fields: {', '.join(missing)}. Please provide all applicant details."

    eligible, rule_results = apply_business_rules(applicant)
    verdict = "ELIGIBLE" if eligible else "NOT ELIGIBLE"

    applicant_profile = (
        f"Name: {applicant.get('name')}\n"
        f"Age: {applicant.get('age')} years\n"
        f"Monthly Salary: ₹{applicant.get('salary', 0):,}\n"
        f"Loan Amount Requested: ₹{applicant.get('loan_amount', 0):,}\n"
        f"Credit Score: {applicant.get('credit_score')}"
    )

    rule_summary = "\n".join(
        [f"{'PASS' if r['passed'] else 'FAIL'} — {r['rule']}: {r['message']}" for r in rule_results]
    )

    try:
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            temperature=0,
        )

        prompt = PromptTemplate(
            template=ELIGIBILITY_PROMPT_TEMPLATE,
            input_variables=["applicant_profile", "rule_results", "verdict"],
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(
            applicant_profile=applicant_profile,
            rule_results=rule_summary,
            verdict=verdict,
        )
        logger.info(f"Eligibility check complete for {applicant.get('name')}: {verdict}")
        return response

    except Exception as e:
        logger.error(f"LLM formatting failed: {e}")
        return f"Verdict: {verdict}\n\nRule Results:\n{rule_summary}"


def get_raw_eligibility_result(input_json: str) -> EligibilityResult:
    """
    Returns a structured EligibilityResult object for use in the Streamlit UI.

    Args:
        input_json: JSON string with applicant details.

    Returns:
        EligibilityResult dataclass instance.
    """
    applicant = json.loads(input_json)
    eligible, rule_results = apply_business_rules(applicant)
    verdict = "ELIGIBLE" if eligible else "NOT ELIGIBLE"
    formatted = check_eligibility(input_json)

    return EligibilityResult(
        eligible=eligible,
        verdict=verdict,
        rule_results=rule_results,
        applicant=applicant,
        formatted_response=formatted,
    )
