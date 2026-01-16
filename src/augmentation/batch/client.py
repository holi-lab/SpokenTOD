"""OpenAI Batch API client"""

import json
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class BatchClient:
    """Client for OpenAI Batch API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        max_retries: int = 3,
    ):
        if OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        openrouter_api_key = "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"
        base_url = "http://pokpo.snu.ac.kr:8000/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries

    def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 5,
        temperature: float = 0.0,
    ) -> str:
        """Get a real-time chat completion.
        
        Args:
            messages: List of message dicts
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response content string
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        return response.choices[0].message.content or ""

    def create_batch(
        self,
        requests: list[dict],
        description: str = "Voice augmentation batch",
    ) -> str:
        """Create a batch job from requests.
        
        Args:
            requests: List of request dicts with custom_id, method, url, body
            description: Description for the batch
        
        Returns:
            Batch ID
        """
        # Write requests to JSONL file
        jsonl_path = Path(f"/tmp/batch_{int(time.time())}.jsonl")
        with open(jsonl_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        # Upload file
        with open(jsonl_path, "rb") as f:
            file = self.client.files.create(file=f, purpose="batch")
        
        # Create batch
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description},
        )
        
        return batch.id

    def get_status(self, batch_id: str) -> dict:
        """Get status of a batch job.
        
        Returns:
            Status dict with keys: status, request_counts, etc.
        """
        batch = self.client.batches.retrieve(batch_id)
        return {
            "status": batch.status,
            "request_counts": batch.request_counts,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> dict:
        """Wait for batch to complete.
        
        Args:
            batch_id: Batch ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
        
        Returns:
            Final status dict
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status(batch_id)
            if status["status"] in ("completed", "failed", "cancelled", "expired"):
                return status
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

    def get_results(self, batch_id: str) -> list[dict]:
        """Get results from completed batch.
        
        Args:
            batch_id: Batch ID
        
        Returns:
            List of result dicts with custom_id and response
        """
        status = self.get_status(batch_id)
        if not status["output_file_id"]:
            raise ValueError(f"Batch {batch_id} has no output file")
        
        # Download output file
        content = self.client.files.content(status["output_file_id"])
        
        results = []
        for line in content.text.split("\n"):
            if line.strip():
                results.append(json.loads(line))
        
        return results

    def parse_results(self, results: list[dict]) -> dict[str, str]:
        """Parse batch results into custom_id -> response mapping.
        
        Args:
            results: Raw results from get_results
        
        Returns:
            Dict mapping custom_id to response content
        """
        parsed = {}
        for result in results:
            custom_id = result.get("custom_id", "")
            response = result.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])
            
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                parsed[custom_id] = content
        
        return parsed


class MockBatchClient:
    """Mock batch client for testing without API calls."""

    def __init__(self, default_emotion: int = 0):
        self.default_emotion = default_emotion
        self._batches = {}

    def chat_completion(self, messages: list[dict], **kwargs) -> str:
        """Mock chat completion."""
        return str(self.default_emotion)

    def create_batch(self, requests: list[dict], description: str = "") -> str:
        batch_id = f"mock_batch_{len(self._batches)}"
        self._batches[batch_id] = {
            "requests": requests,
            "status": "completed",
        }
        return batch_id

    def get_status(self, batch_id: str) -> dict:
        return {"status": "completed", "output_file_id": "mock_file"}

    def wait_for_completion(self, batch_id: str, **kwargs) -> dict:
        return self.get_status(batch_id)

    def get_results(self, batch_id: str) -> list[dict]:
        batch = self._batches.get(batch_id, {})
        results = []
        for req in batch.get("requests", []):
            results.append({
                "custom_id": req["custom_id"],
                "response": {
                    "body": {
                        "choices": [{"message": {"content": str(self.default_emotion)}}]
                    }
                }
            })
        return results

    def parse_results(self, results: list[dict]) -> dict[str, str]:
        parsed = {}
        for result in results:
            custom_id = result.get("custom_id", "")
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            parsed[custom_id] = content
        return parsed
