from dotenv import load_dotenv

load_dotenv()

from graph.graph import app


def main():
    print("Hello from agenticrag-langgraph!")
    while True:
        question: str = input("Ask a question: \n")
        response = app.invoke(input={"question": question})
        print("---RESPONSE---")
        print(response["generation"])
        print("---DOCUMENTS---")
        print(response["documents"])


if __name__ == "__main__":
    main()
