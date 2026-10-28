import os

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")
    print(f"LANGSMITH_API_KEY: {os.getenv('LANGSMITH_API_KEY')}")
