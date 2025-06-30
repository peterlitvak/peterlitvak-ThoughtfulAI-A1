"""
Mini RAG Backend implementation using LangChain (latest stable) with an in‑memory ChromaDB
vector store and per‑session chat history.

Assumptions
-----------
* LangChain v0.2+ is available.
* The OpenAI and ChromaDB community integrations are installed
* `knowledge_base_path` points to a JSON file structured as a list of objects, each
  containing at least a `text` field (and optionally an `id`).
* Only small, static KB files are expected, so the whole corpus is loaded into
  memory at start‑up.
* A simple `InMemoryChatHistoryProvider` is used to track history keyed by
  `session_id`.

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
    """
    Provides in-memory storage for chat history, organized by session ID.
    Useful for maintaining conversation context during a session.
    Not persistent across process restarts.

    Methods
    -------
    get_history(session_id: str) -> List[ChatMessage]
        Retrieve the chat history for a given session.
    append(session_id: str, message: ChatMessage) -> None
        Append a message to the session's chat history.
    clear(session_id: str) -> None
        Clear the chat history for a session.
    """
    def __init__(self) -> None:
        """
        Initialize the in-memory chat history provider.
        """
        self._sessions: Dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """
        Retrieve the chat history for a given session.

        Parameters
        ----------
        session_id : str
            The unique identifier for the chat session.

        Returns
        -------
        List[ChatMessage]
            The list of chat messages for the session.
        """
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, message: ChatMessage) -> None:
        """
        Append a message to the session's chat history.

        Parameters
        ----------
        session_id : str
            The unique identifier for the chat session.
        message : ChatMessage
            The message to append.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(message)

    def clear(self, session_id: str) -> None:
        """
        Clear the chat history for a session.

        Parameters
        ----------
        session_id : str
            The unique identifier for the chat session.
        """
        self._sessions[session_id] = []


@dataclass
class RAGService:
    """
    Lightweight Retrieval-Augmented Generation (RAG) service.

    Loads a knowledge base from a JSON file and builds a ChromaDB vector index.
    Tracks per-session chat history. Uses an LLM (OpenAI) for generating answers,
    with streaming support. Injects a system prompt and chat context into every LLM call.

    Parameters
    ----------
    system_prompt : str
        The system prompt injected into every LLM call.
    knowledge_base_path : Union[str, Path]
        Path to the JSON knowledge base file.
    chat_memory_provider : InMemoryChatHistoryProvider
        Provider for storing per-session chat history.
    llm_api_key : str
        OpenAI API key for LLM usage.
    llm : ChatOpenAI, optional
        Pre-initialized LLM instance (if not provided, will be created).
    """
    system_prompt: str
    knowledge_base_path: Union[str, Path]
    chat_memory_provider: InMemoryChatHistoryProvider
    llm_api_key: str
    llm: ChatOpenAI = None
    embeddings: OpenAIEmbeddings = field(init=False)
    vectordb: Chroma = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize embeddings, vector DB, and LLM if not provided. Ensures OpenAI API key is set.
        """
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

    def _load_knowledge_base(self, path: Union[str, Path]) -> Chroma:
        """
        Load documents from *path* and build a ChromaDB index in memory.
        Raises ValueError if file is missing, not JSON, or structure is invalid.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Knowledge base file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Knowledge base file is not valid JSON: {e}")

        if isinstance(data, dict) and "questions" in data:
            docs = data["questions"]
            if not isinstance(docs, list):
                raise ValueError("Expected 'questions' to be a list in the knowledge base.")
            for doc in docs:
                if not isinstance(doc, dict) or "question" not in doc or "answer" not in doc:
                    raise ValueError("Each item in 'questions' must have 'question' and 'answer' fields.")
            texts = [f"Q: {doc['question']}\nA: {doc['answer']}" for doc in docs]
            metadatas = [{"id": idx} for idx, _ in enumerate(docs)]
        elif isinstance(data, list):
            for doc in data:
                if not isinstance(doc, dict) or "text" not in doc:
                    raise ValueError("Each item in the KB list must have a 'text' field.")
            texts = [doc["text"] for doc in data]
            metadatas = [{"id": doc.get("id", idx)} for idx, doc in enumerate(data)]
        else:
            raise ValueError("Knowledge base format not recognized. Must be a list or dict with 'questions' key.")

        if not texts:
            raise ValueError("Knowledge base is empty after parsing.")

        return Chroma.from_texts(texts, self.embeddings, metadatas=metadatas)

    def _retrieve(self, query: str, k: int = 4) -> str:
        """
        Return the top-k passages concatenated with separators.

        Parameters
        ----------
        query : str
            The user query for retrieval.
        k : int, optional
            Number of top passages to retrieve (default=4).

        Returns
        -------
        str
            Concatenated relevant passages from the knowledge base.
        """
        docs = self.vectordb.similarity_search(query, k=k)
        return "\n---\n".join(d.page_content for d in docs)

    def _compile_history(self, history: List[ChatMessage]) -> str:
        """
        Compile the chat history into a formatted string for prompt injection.

        Parameters
        ----------
        history : List[ChatMessage]
            List of chat messages in the session.

        Returns
        -------
        str
            Formatted conversation history string.
        """
        return "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in history
        )

    def chat(
        self,
        user_query: str,
        session_id: str,
        *,
        stream: bool = True,
        top_k: int = 4,
    ) -> Union[str, Iterable[str]]:
        """
        Answer a user query within the context of a session, using retrieval-augmented generation.

        Parameters
        ----------
        user_query : str
            The user's question.
        session_id : str
            Unique identifier for the chat session.
        stream : bool, optional
            If True, yields answer chunks as they arrive (default=True).
        top_k : int, optional
            Number of passages to retrieve from the KB (default=4).

        Returns
        -------
        Union[str, Iterable[str]]
            The answer as a string or a generator yielding streamed answer chunks.
        """
        history = self.chat_memory_provider.get_history(session_id)
        context = self._retrieve(user_query, k=top_k)
        prompt = self._qa_prompt.format(
            system_prompt=self.system_prompt,
            context=context,
            history=self._compile_history(history),
            question=user_query,
        )
        self.chat_memory_provider.append(session_id, HumanMessage(content=user_query))
        if stream:
            return self._stream_response(prompt, session_id)
        else:
            answer = self.llm.invoke(prompt).content
            self.chat_memory_provider.append(session_id, AIMessage(content=answer))
            return answer

    def _stream_response(
        self, prompt: str, session_id: str
    ) -> Generator[str, None, None]:
        """
        Internal streaming helper. Streams LLM answer chunks and appends the final response
        to chat history.

        Parameters
        ----------
        prompt : str
            The full prompt to send to the LLM.
        session_id : str
            Unique identifier for the chat session.

        Yields
        ------
        str
            Chunks of the LLM-generated answer as they arrive.
        """
        collected = ""
        for chunk in self.llm.stream(prompt):
            chunk_str = getattr(chunk, "content", None)
            if chunk_str is None:
                chunk_str = str(chunk)
            collected += chunk_str
            yield chunk_str
        self.chat_memory_provider.append(session_id, AIMessage(content=collected))
