import os

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI
)


load_dotenv()


class GeminiLLM:
    """
    Gemini LLM wrapper.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        temperature: float = 0
    ):
        self.model_name = os.getenv("GEMINI_MODEL", model_name)
        self.temperature = temperature

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found "
                "in environment variables."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=api_key
        )

        print(
            f"Gemini initialized: "
            f"{self.model_name}"
        )

    def generate(
        self,
        prompt: str
    ) -> str:
        """Generate a response from Gemini."""

        response = self.llm.invoke(prompt)
        content = response.content

        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            text_blocks = []
            for item in content:
                if isinstance(item, str):
                    text_blocks.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_blocks.append(item["text"])
            return "\n".join(text_blocks).strip()

        return str(content).strip()