from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(temperature=0)


class GradeHallucatinations(BaseModel):
    """Binary score for hallucinations present in generated answer"""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucatinations)


system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of documents.
    Give a binary score of 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n{documents} \n\n LLM generation: {generation}")
    ]
)

hallucination_grader: RunnableSerializable = hallucination_prompt | structured_llm_grader