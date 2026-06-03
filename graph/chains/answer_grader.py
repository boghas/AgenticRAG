from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):
    """Binary score for answer addressing the question."""

    binary_score: bool = Field(
        description="Answer addresses the question 'yes' or 'no'"
    )


llm = ChatOpenAI(temperature=0)
structured_llm = llm.with_structured_output(GradeAnswer)


system = """You are a grader assessing whether an answer addresses / resolves a question \n.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""


grade_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader_chain: RunnableSerializable = grade_answer_prompt | structured_llm