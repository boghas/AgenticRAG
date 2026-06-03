from dotenv import load_dotenv

load_dotenv()

from graph.chains.router import question_router_chain, RouteQuery
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.hallucinations_grader import hallucination_grader, GradeHallucatinations
from graph.chains.generation import generation_chain
from ingestion import retriever

from pprint import pprint


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "Make pizza"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucinations_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})

    is_hallucinated: GradeHallucatinations = hallucination_grader.invoke({"documents": docs, "generation": generation})

    assert is_hallucinated.binary_score


def test_hallucinations_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    is_hallucinated: GradeHallucatinations = hallucination_grader.invoke(
        {"documents": docs, "generation": "In order to make pizza we need to first start with the dough."}
    )

    assert not is_hallucinated.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res: RouteQuery = question_router_chain.invoke(
        {"question": question}
    )

    assert res.datasource.lower() == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"
    res: RouteQuery = question_router_chain.invoke(
        {"question": question}
    )

    assert res.datasource.lower() == "websearch"