"""
Mini RAG Backend implementation using LangChain (latest stable) with an in‑memory ChromaDB vector store and per‑session chat history.

Assumptions
-----------
* LangChain v0.2+ is available.
* The OpenAI and ChromaDB community integrations are installed:
    pip install langchain langchain-openai langchain-community chromadb
* `knowledge_base_path` points to a JSON file structured as a list of objects, each
  containing at least a `text` field (and optionally an `id`).
* Only small, static KB files are expected, so the whole corpus is loaded into
  memory at start‑up.
* A simple `InMemoryChatHistoryProvider` is used to track history keyed by
  `session_id`. Swap this for a redis/db‑backed provider if you need
  persistence.

Usage example (non‑streaming)::

    from rag_backend import InMemoryChatHistoryProvider, RAGService
    from langchain_openai import ChatOpenAI

    history = InMemoryChatHistoryProvider()
    rag = RAGService(
        system_prompt="You are a helpful customer‑support assistant.",
        knowledge_base_path="data.json",
        chat_memory_provider=history,
        llm=ChatOpenAI(model_name="gpt-4o-mini"),
        llm_api_key="sk‑...",
    )

    answer = rag.chat("What is Thoughtful AI?", session_id="abc123", stream=False)
    print(answer)

For streaming, iterate over the generator returned when `stream=True`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Union, Generator

from langchain.schema import AIMessage, ChatMessage, HumanMessage
from langchain.prompts import PromptTemplate

# LangChain v0.2 splits integrations into separate packages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
from langchain.vectorstores import Chroma  # type: ignore


class InMemoryChatHistoryProvider:
    """Very simple in‑memory store for per‑session chat history."""

    def __init__(self) -> None:
        self._sessions: Dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str) -> List[ChatMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, message: ChatMessage) -> None:
        self._sessions.setdefault(session_id, []).append(message)

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


@dataclass
class RAGService:
    """Lightweight RAG service with embedded KB and chat‑history tracking."""

    system_prompt: str
    knowledge_base_path: Union[str, Path]
    chat_memory_provider: InMemoryChatHistoryProvider
    llm_api_key: str
    llm: ChatOpenAI = None
    embeddings: OpenAIEmbeddings = field(init=False)
    vectordb: Chroma = field(init=False)

    def __post_init__(self) -> None:
        # Configure API key early so downstream components pick it up
        os.environ.setdefault("OPENAI_API_KEY", self.llm_api_key)

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = self._load_knowledge_base(self.knowledge_base_path)

        if self.llm is None:
            self.llm = ChatOpenAI(openai_api_key=self.llm_api_key, streaming=True)

        self._qa_prompt = PromptTemplate(
            template=(
                "{system_prompt}\n\n"
                "CONTEXT:\n{context}\n\n"
                "Conversation so far:\n{history}\n\n"
                "User: {question}\nAI:"
            ),
            input_variables=["system_prompt", "context", "history", "question"],
        )

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------

    def _load_knowledge_base(self, path: Union[str, Path]) -> Chroma:
        """Load documents from *path* and build a ChromaDB index in memory."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both list and dict with 'questions' key
        if isinstance(data, dict) and "questions" in data:
            docs = data["questions"]
            # Combine question and answer into a single text field
            texts = [f"Q: {doc['question']}\nA: {doc['answer']}" for doc in docs]
            metadatas = [{"id": idx} for idx, _ in enumerate(docs)]
        else:
            texts = [doc["text"] for doc in data]
            metadatas = [{"id": doc.get("id", idx)} for idx, doc in enumerate(data)]

        return Chroma.from_texts(texts, self.embeddings, metadatas=metadatas)

    def _retrieve(self, query: str, k: int = 4) -> str:
        """Return the top‑*k* passages concatenated with separators."""
        docs = self.vectordb.similarity_search(query, k=k)
        return "\n---\n".join(d.page_content for d in docs)

    def _compile_history(self, history: List[ChatMessage]) -> str:
        return "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in history
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def chat(
        self,
        user_query: str,
        session_id: str,
        *,
        stream: bool = True,
        top_k: int = 4,
    ) -> Union[str, Iterable[str]]:
        """Answer *user_query* within the context of *session_id*.

        If *stream* is True, returns a generator yielding chunks as they arrive.
        Otherwise, returns the full response string.
        """

        history = self.chat_memory_provider.get_history(session_id)
        context = self._retrieve(user_query, k=top_k)

        prompt = self._qa_prompt.format(
            system_prompt=self.system_prompt,
            context=context,
            history=self._compile_history(history),
            question=user_query,
        )

        # 1️⃣ Add the user's message to history NOW so retrieval for the next turn
        #    can include it if desired.
        self.chat_memory_provider.append(session_id, HumanMessage(content=user_query))

        if stream:
            return self._stream_response(prompt, session_id)
        else:
            answer = self.llm.invoke(prompt).content  # synchronous call
            self.chat_memory_provider.append(session_id, AIMessage(content=answer))
            return answer

    # ------------------------------------------------------------------
    # Internal streaming helper
    # ------------------------------------------------------------------

    def _stream_response(
        self, prompt: str, session_id: str
    ) -> Generator[str, None, None]:
        collected = ""
        for chunk in self.llm.stream(prompt):
            # Convert chunk to string if it's not already
            chunk_str = getattr(chunk, "content", None)
            if chunk_str is None:
                chunk_str = str(chunk)
            collected += chunk_str
            yield chunk_str
        # Store final AI response in history *after* streaming completes
        self.chat_memory_provider.append(session_id, AIMessage(content=collected))
