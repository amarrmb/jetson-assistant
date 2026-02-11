"""
LLM integration for voice assistant.

Supports:
- Ollama (local LLMs: Llama, Mistral, Phi, etc.)
- OpenAI API (GPT-4, GPT-3.5)
- Anthropic API (Claude)
- Custom HTTP endpoints
"""

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolCallResult:
    """A tool call returned by the LLM."""

    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Response from LLM."""

    text: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    tool_calls: list[ToolCallResult] = field(default_factory=list)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(
        self, prompt: str, context: Optional[list[dict]] = None, images: Optional[list[str]] = None
    ) -> LLMResponse:
        """
        Generate a response to the user's prompt.

        Args:
            prompt: User's input text
            context: Optional conversation history
            images: Optional list of base64-encoded images (for VLMs)

        Returns:
            LLMResponse with generated text
        """
        pass

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the assistant."""
        self.system_prompt = prompt

    def classify_intent(self, text: str, prompt: str) -> Optional[str]:
        """Fast intent classification. Returns category string or None."""
        return None


class OllamaLLM(LLMBackend):
    """
    Local LLM using Ollama.

    Ollama must be running: `ollama serve`
    Models: llama3.2, mistral, phi3, gemma2, etc.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful voice assistant. Keep answers concise (1-2 sentences). Be direct and knowledgeable."""

    RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant. Use the provided context to give accurate answers. Keep answers under 30 words but make them insightful."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
        rag_collection: Optional[str] = None,  # Enable RAG with collection name
    ):
        """
        Initialize Ollama LLM.

        Args:
            model: Ollama model name (e.g., "llama3.2:3b", "mistral", "phi3")
            host: Ollama server URL
            system_prompt: Custom system prompt
            rag_collection: RAG collection name (enables RAG if set)
        """
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "Ollama Python client not installed. "
                "Install with: pip install ollama"
            ) from e

        self.model = model
        self.host = host
        self._client = ollama.Client(host=host)

        # Initialize RAG if collection specified
        self.rag = None
        if rag_collection:
            try:
                from jetson_speech.rag import RAGPipeline
                self.rag = RAGPipeline(collection_name=rag_collection)
                if self.rag.count() > 0:
                    print(f"RAG enabled: {rag_collection} ({self.rag.count()} chunks)", file=sys.stderr)
                    self.system_prompt = system_prompt or self.RAG_SYSTEM_PROMPT
                else:
                    print(f"RAG collection '{rag_collection}' is empty. Run build script first.", file=sys.stderr)
                    self.rag = None
                    self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
            except ImportError as e:
                print(f"RAG not available: {e}", file=sys.stderr)
                self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        else:
            self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Check if model is available
        try:
            self._client.show(model)
            print(f"Ollama LLM ready: {model}", file=sys.stderr)
        except Exception:
            print(f"Model '{model}' not found. Pulling...", file=sys.stderr)
            self._client.pull(model)
            print(f"Ollama LLM ready: {model}", file=sys.stderr)

    def check_condition(self, prompt: str, image_b64: str) -> bool:
        """
        Fast binary VLM query for vision monitoring.

        Sends a yes/no question with an image and returns True if the
        model responds with YES. Uses minimal token budget for speed.

        Args:
            prompt: Yes/no question (e.g., "Do you see an apple?")
            image_b64: Base64-encoded JPEG image

        Returns:
            True if VLM responds YES, False otherwise.
        """
        import time

        messages = [
            {"role": "system", "content": "Respond with ONLY the word YES or NO."},
            {"role": "user", "content": prompt, "images": [image_b64]},
        ]

        start = time.perf_counter()
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={"num_predict": 5, "temperature": 0},
            )
            latency = (time.perf_counter() - start) * 1000
            answer = response["message"]["content"].strip().upper()
            print(f"  [VisionMonitor check: {answer} ({latency:.0f}ms)]", file=sys.stderr)
            return answer.startswith("YES")
        except Exception as e:
            print(f"  [VisionMonitor check error: {e}]", file=sys.stderr)
            return False

    def generate(
        self, prompt: str, context: Optional[list[dict]] = None, images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """Generate response using Ollama."""
        import time

        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation context
        if context:
            messages.extend(context)

        # For vision, inline brevity instruction — VLMs ignore system prompts
        user_content = prompt
        if images:
            user_content = f"BRIEF answer, max 15 words. {prompt}"

        user_message = {"role": "user", "content": user_content}
        if images:
            user_message["images"] = images
        messages.append(user_message)

        num_predict = 40 if images else 25

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "options": {"num_predict": num_predict},
        }
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        response = self._client.chat(**kwargs)
        latency = (time.perf_counter() - start) * 1000

        # Parse tool calls from response
        tool_calls = []
        for tc in response.get("message", {}).get("tool_calls", []):
            func = tc.get("function", {})
            tool_calls.append(ToolCallResult(
                name=func.get("name", ""),
                arguments=func.get("arguments", {}),
            ))

        return LLMResponse(
            text=response["message"].get("content", "") or "",
            model=self.model,
            tokens_used=response.get("eval_count", 0),
            latency_ms=latency,
            tool_calls=tool_calls,
        )

    def generate_stream(
        self, prompt: str, context: Optional[list[dict]] = None, images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
    ):
        """
        Stream response from Ollama, yielding complete sentences or ToolCallResult.

        Yields:
            str | ToolCallResult: Sentences as they complete, then any tool calls
        """
        import re

        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.extend(context)

        # Add RAG context if available
        user_content = prompt
        if self.rag is not None:
            rag_context = self.rag.build_context(prompt, top_k=3)
            user_content = f"Context:\n{rag_context}\n\nQuestion: {prompt}"

        # For vision, inline brevity instruction — VLMs ignore system prompts
        if images:
            user_content = f"BRIEF answer, max 15 words. {user_content}"

        user_message = {"role": "user", "content": user_content}
        if images:
            user_message["images"] = images
        messages.append(user_message)

        num_predict = 40 if images else 60

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": num_predict},
        }
        if tools:
            kwargs["tools"] = tools

        buffer = ""
        # Match sentence endings: period, !, ?, or colon followed by space/newline
        sentence_pattern = re.compile(r'[.!?:]\s+')
        pending_tool_calls = []

        try:
            for chunk in self._client.chat(**kwargs):
                # Check if stream is done — Ollama puts tool_calls in the final chunk
                if chunk.get("done", False):
                    for tc in chunk.get("message", {}).get("tool_calls", []):
                        func = tc.get("function", {})
                        pending_tool_calls.append(ToolCallResult(
                            name=func.get("name", ""),
                            arguments=func.get("arguments", {}),
                        ))
                    break

                content = chunk.get("message", {}).get("content", "")
                if not content:
                    continue

                buffer += content

                # Check for complete sentences - yield as soon as we find one
                match = sentence_pattern.search(buffer)
                if match:
                    end_pos = match.end()
                    sentence = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:].strip()
                    if sentence and len(sentence) > 2:  # Allow short sentences like "No."
                        yield sentence

        except Exception as e:
            print(f"LLM stream error: {e}", file=sys.stderr)

        # Yield any remaining text (even very short responses like "Blue")
        if buffer.strip():
            yield buffer.strip()

        # Yield tool calls after all text
        for tc in pending_tool_calls:
            yield tc


class OpenAILLM(LLMBackend):
    """
    LLM using OpenAI API (GPT-4, GPT-3.5).

    Requires OPENAI_API_KEY environment variable.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful voice assistant.
Keep responses concise (1-3 sentences). Speak naturally without formatting."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize OpenAI LLM.

        Args:
            model: OpenAI model name
            api_key: API key (or set OPENAI_API_KEY env)
            system_prompt: Custom system prompt
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI client not installed. "
                "Install with: pip install openai"
            ) from e

        import os

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._client = OpenAI(api_key=api_key)

        print(f"OpenAI LLM ready: {model}", file=sys.stderr)

    def generate(
        self, prompt: str, context: Optional[list[dict]] = None, images: Optional[list[str]] = None
    ) -> LLMResponse:
        """Generate response using OpenAI."""
        import time

        if images:
            print("Warning: OpenAI vision not implemented, images ignored", file=sys.stderr)

        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.extend(context)

        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=150,
        )
        latency = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency,
        )


class AnthropicLLM(LLMBackend):
    """
    LLM using Anthropic API (Claude).

    Requires ANTHROPIC_API_KEY environment variable.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful voice assistant.
Keep responses concise (1-3 sentences). Speak naturally without formatting."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Anthropic LLM.

        Args:
            model: Anthropic model name
            api_key: API key (or set ANTHROPIC_API_KEY env)
            system_prompt: Custom system prompt
        """
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic client not installed. "
                "Install with: pip install anthropic"
            ) from e

        import os

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._client = anthropic.Anthropic(api_key=api_key)

        print(f"Anthropic LLM ready: {model}", file=sys.stderr)

    def generate(
        self, prompt: str, context: Optional[list[dict]] = None, images: Optional[list[str]] = None
    ) -> LLMResponse:
        """Generate response using Anthropic."""
        import time

        if images:
            print("Warning: Anthropic vision not implemented, images ignored", file=sys.stderr)

        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            system=self.system_prompt,
            messages=messages,
            max_tokens=150,
        )
        latency = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=response.content[0].text,
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency,
        )


class VLLMLLM(LLMBackend):
    """
    VLM/LLM using vLLM's OpenAI-compatible API.

    Supports vision (image inputs) via the OpenAI multimodal content format.
    Designed for local vLLM serving on Jetson with models like Qwen2.5-VL.

    Start vLLM server:
        docker run --runtime=nvidia --network host \\
            nvcr.io/nvidia/vllm:25.12.post1-py3 \\
            vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful voice assistant with vision. "
        "Keep responses concise (1-2 sentences). "
        "Speak naturally. Be direct and accurate."
    )

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        host: str = "http://localhost:8000/v1",
        system_prompt: Optional[str] = None,
        **_kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI client not installed. Install with: pip install openai"
            ) from e

        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._client = OpenAI(api_key="not-needed", base_url=host)

        # Quick health check
        try:
            models = self._client.models.list()
            available = [m.id for m in models.data]
            if model not in available and available:
                print(f"vLLM models available: {available}", file=sys.stderr)
                # Use first available model if specified one isn't found
                if len(available) == 1:
                    self.model = available[0]
                    print(f"Using: {self.model}", file=sys.stderr)
            print(f"vLLM ready: {self.model} at {host}", file=sys.stderr)
        except Exception as e:
            print(f"vLLM warning: could not reach {host}: {e}", file=sys.stderr)
            print("Make sure vLLM server is running.", file=sys.stderr)

    @staticmethod
    def _build_content(text: str, images: Optional[list[str]] = None) -> list[dict] | str:
        """Build OpenAI-format multimodal content array."""
        if not images:
            return text

        content = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        content.append({"type": "text", "text": text})
        return content

    def check_condition(self, prompt: str, image_b64: str) -> bool:
        """Fast binary VLM query for vision monitoring."""
        import time

        messages = [
            {"role": "system", "content": "Respond with ONLY the word YES or NO."},
            {
                "role": "user",
                "content": self._build_content(prompt, [image_b64]),
            },
        ]

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=5,
                temperature=0,
            )
            latency = (time.perf_counter() - start) * 1000
            answer = response.choices[0].message.content.strip().upper()
            print(f"  [VisionMonitor check: {answer} ({latency:.0f}ms)]", file=sys.stderr)
            return answer.startswith("YES")
        except Exception as e:
            print(f"  [VisionMonitor check error: {e}]", file=sys.stderr)
            return False

    def classify_intent(self, text: str, prompt: str) -> Optional[str]:
        """Fast intent classification (~90ms). Returns category string."""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=5,
                temperature=0,
            )
            return response.choices[0].message.content.strip().split()[0].upper()
        except Exception:
            return None

    def generate(
        self, prompt: str, context: Optional[list[dict]] = None, images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """Generate response using vLLM."""
        import time

        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.extend(context)

        user_content = prompt
        if images:
            user_content = f"BRIEF answer, max 15 words. {prompt}"

        messages.append({
            "role": "user",
            "content": self._build_content(user_content, images),
        })

        max_tokens = 60 if images else 150

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000

        # Parse tool calls from response
        tool_calls = []
        msg = response.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                tool_calls.append(ToolCallResult(name=tc.function.name, arguments=args))

        return LLMResponse(
            text=msg.content or "",
            model=self.model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency,
            tool_calls=tool_calls,
        )

    def generate_stream(
        self, prompt: str, context: Optional[list[dict]] = None, images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
    ):
        """Stream response from vLLM, yielding complete sentences or ToolCallResult.

        Yields:
            str | ToolCallResult: Sentences as they complete, then any tool calls
        """
        import re

        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.extend(context)

        user_content = prompt
        if images:
            user_content = f"BRIEF answer, max 15 words. {user_content}"

        messages.append({
            "role": "user",
            "content": self._build_content(user_content, images),
        })

        max_tokens = 60 if images else 150

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = tools

        buffer = ""
        sentence_pattern = re.compile(r'[.!?:]\s+')
        # Accumulate tool call deltas: {index: {name, arguments_str}}
        tool_calls_acc: dict[int, dict] = {}

        try:
            stream = self._client.chat.completions.create(**kwargs)

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Accumulate streamed tool call chunks
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"name": "", "arguments": ""}
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_acc[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

                content = delta.content if delta and delta.content else ""
                if not content:
                    continue

                buffer += content

                match = sentence_pattern.search(buffer)
                if match:
                    end_pos = match.end()
                    sentence = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:].strip()
                    if sentence and len(sentence) > 2:
                        yield sentence

        except Exception as e:
            print(f"vLLM stream error: {e}", file=sys.stderr)

        if buffer.strip():
            yield buffer.strip()

        # Yield accumulated tool calls after all text
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            if tc["name"]:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                yield ToolCallResult(name=tc["name"], arguments=args)


class SimpleLLM(LLMBackend):
    """
    Simple rule-based "LLM" for testing without real LLM.

    Handles basic commands and responds with canned phrases.
    """

    RESPONSES = {
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! What can I do for you?",
        "how are you": "I'm doing great, thanks for asking!",
        "what time is it": "I'm sorry, I don't have access to the current time.",
        "tell me a joke": "Why do programmers prefer dark mode? Because light attracts bugs!",
        "thank you": "You're welcome!",
        "thanks": "Happy to help!",
        "goodbye": "Goodbye! Have a great day!",
        "bye": "See you later!",
    }

    def __init__(self):
        self.system_prompt = ""
        print("Using simple rule-based responses (no LLM)", file=sys.stderr)

    def generate(
        self, prompt: str, context: Optional[list[dict]] = None, images: Optional[list[str]] = None
    ) -> LLMResponse:
        """Generate response using simple rules."""
        prompt_lower = prompt.lower().strip()

        # Check for exact matches
        for key, response in self.RESPONSES.items():
            if key in prompt_lower:
                return LLMResponse(text=response, model="simple", tokens_used=0)

        # Default response
        return LLMResponse(
            text="I'm not sure how to respond to that. Try asking me something else!",
            model="simple",
            tokens_used=0,
        )


def create_llm(
    backend: str = "ollama",
    model: Optional[str] = None,
    **kwargs,
) -> LLMBackend:
    """
    Factory function to create LLM backend.

    Args:
        backend: "ollama", "openai", "anthropic", or "simple"
        model: Model name (backend-specific)
        **kwargs: Backend-specific options

    Returns:
        LLMBackend instance
    """
    if backend == "ollama":
        return OllamaLLM(model=model or "llama3.2:3b", **kwargs)
    elif backend == "vllm":
        return VLLMLLM(model=model or "Qwen/Qwen2.5-VL-7B-Instruct", **kwargs)
    elif backend == "openai":
        return OpenAILLM(model=model or "gpt-4o-mini", **kwargs)
    elif backend == "anthropic":
        return AnthropicLLM(model=model or "claude-3-haiku-20240307", **kwargs)
    elif backend == "simple":
        return SimpleLLM()
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
