import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class GeminiLLM:
    """
    Gemini LLM wrapper with automatic model fallbacks and rate-limit handling.
    """

    FALLBACK_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-pro",
    ]

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        temperature: float = 0
    ):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables."
            )

        self.model_name = os.getenv("GEMINI_MODEL", model_name)
        self.temperature = temperature
        self.llm = self._create_llm(self.model_name)

        print(f"Gemini initialized: {self.model_name}")

    def _create_llm(self, model: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=self.temperature,
            google_api_key=self.api_key
        )

    def generate(
        self,
        prompt: str
    ) -> str:
        """Generate a response from Gemini with fallback models on quota/rate-limit errors."""

        models_to_try = [self.model_name] + [m for m in self.FALLBACK_MODELS if m != self.model_name]

        last_error = None

        for model in models_to_try:
            try:
                llm = self._create_llm(model) if model != self.model_name else self.llm
                response = llm.invoke(prompt)
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

            except Exception as e:
                err_msg = str(e)
                last_error = e
                # Check for 429 Quota/Rate limit or 404 Model Not Found
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "404" in err_msg or "NOT_FOUND" in err_msg:
                    print(f"[!] Warning: Model '{model}' failed ({'Quota/Rate limit reached' if '429' in err_msg or 'RESOURCE_EXHAUSTED' in err_msg else 'Not found'}). Trying fallback model...")
                    continue
                else:
                    raise e

        # If all models failed due to quota limits
        return (
            "⚠️ API Quota Limit Exceeded: The Gemini Free Tier daily limit (20 requests/day per model) "
            "has been reached for your API key.\n\n"
            "Options to fix this:\n"
            "1. Wait for the quota reset period to expire.\n"
            "2. Use a different Gemini API Key in your .env file.\n"
            "3. Upgrade your Google AI Studio key billing settings."
        )