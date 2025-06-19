# custom_llm.py
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
import requests
import json

class CustomMistralLLM(LLM):
    endpoint: str
    auth_token: str
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "custom_mistral"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        data = {
            "prompt": prompt,
        }
        response = requests.post(self.endpoint, headers=headers, json=data, verify=False)
        response.raise_for_status()

        response_json = response.json()
        if "outputs" in response_json and len(response_json["outputs"]) > 0:
            return response_json["outputs"][0]["text"].strip()
        else:
            return "No text generated."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"endpoint": self.endpoint, "temperature": self.temperature}