from dotenv import load_dotenv

load_dotenv()

from graph.graph import app


def main():
    print("Hello from agenticrag-langgraph!")
    print(app.invoke(input={"question": "what is agent memory"}))


if __name__ == "__main__":
    main()
