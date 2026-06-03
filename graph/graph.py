from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph

from graph.chains.router import question_router_chain, RouteQuery
from graph.chains.answer_grader import answer_grader_chain
from graph.chains.hallucinations_grader import hallucination_grader
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState) -> str:
    print("---ASSES GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO THE QUESTION"
        )

        return WEBSEARCH
    
    print("---DECISION: GENERATE---")
    return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucinations_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("--- GRADE GENERATION vs QUESTION---")
        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        
        if answer_grade := score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS")
        return "not supported"
    

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router_chain.invoke({"question": question})

    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    
    print("ROUTE QUESTION TO VECTORSTORE")
    return RETRIEVE

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    path_map={
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE
    }
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={
        "not useful": WEBSEARCH,
        "not supported": GENERATE,
        "useful": END,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")