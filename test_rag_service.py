"""
Integration tests for the RAGService using the real knowledge base and OpenAI LLM.
Covers streaming, non-streaming, knowledge base usage, and chat history.
"""
import os
import unittest
from rag_backend import RAGService, InMemoryChatHistoryProvider
from dotenv import load_dotenv

class TestRAGServiceIntegration(unittest.TestCase):
    """
    Integration tests for RAGService.
    Ensures correct behavior for streaming, retrieval, and chat history.
    """
    @classmethod
    def setUpClass(cls):
        """
        Load environment variables and instantiate a real RAGService for all tests.
        """
        # Load environment variables from .env file
        load_dotenv()
        # Get the OpenAI API key from the environment
        cls.api_key = os.environ.get("OPENAI_API_KEY")
        # Ensure the API key is set for the integration test
        assert cls.api_key, "OPENAI_API_KEY must be set in the environment for integration test."
        # Set the path to the knowledge base file
        cls.kb_path = "data.json"  # Use your real data.json
        # Set the system prompt for the RAGService
        cls.system_prompt = "You are a helpful AI."
        # Create an in-memory chat history provider
        cls.chat_memory = InMemoryChatHistoryProvider()
        # Instantiate the real LLM and embeddings via RAGService
        cls.rag = RAGService(
            system_prompt=cls.system_prompt,
            knowledge_base_path=cls.kb_path,
            chat_memory_provider=cls.chat_memory,
            llm_api_key=cls.api_key,
            llm=None  # Let __post_init__ handle LLM instantiation if supported
        )

    def test_chat_non_streaming(self):
        """
        Test that a non-streaming chat returns a non-empty string answer.
        """
        # Test a non-streaming chat with a simple question
        answer = self.rag.chat("What is Python?", session_id="integration1", stream=False)
        # Ensure the answer is a non-empty string
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        # Print the answer for debugging
        print("Non-streaming answer:", answer)

    def test_chat_streaming(self):
        """
        Test that streaming chat yields at least one non-empty string chunk.
        """
        # Test a streaming chat with a simple question
        chunks = list(self.rag.chat("What is AI?", session_id="integration2", stream=True))
        # Ensure at least one chunk is a non-empty string
        self.assertTrue(any(isinstance(c, str) and c for c in chunks))
        # Print the chunks for debugging
        print("Streaming answer chunks:", chunks)

    def test_knowledge_base_usage(self):
        """
        Test that a question present in the KB yields an answer containing expected KB content.
        """
        # Define a question and expected answer from the knowledge base
        question = "What does the eligibility verification agent (EVA) do?"
        expected_phrase = "EVA automates the process of verifying a patient’s eligibility and benefits information"
        # Test the chat with the question
        answer = self.rag.chat(question, session_id="kb_usage", stream=False)
        # Ensure the answer contains the expected phrase
        self.assertIn(expected_phrase, answer)
        # Print the answer for debugging
        print("Knowledge base answer:", answer)

    def test_chat_history(self):
        """
        Test that chat history is tracked and both user and AI messages are present in order.
        """
        # Define a session ID for the chat history test
        session_id = "history_test"
        # First turn: ask a question and get an answer
        q1 = "What is EVA?"
        a1 = self.rag.chat(q1, session_id=session_id, stream=False)
        # Second turn: ask a follow-up question and get an answer
        q2 = "What does it do?"
        a2 = self.rag.chat(q2, session_id=session_id, stream=False)
        # Get the chat history for the session
        history = self.rag.chat_memory_provider.get_history(session_id)
        # Ensure the history has at least 4 messages (2 user + 2 AI)
        self.assertGreaterEqual(len(history), 4)
        # Ensure the first message is the user's question
        self.assertIn(q1, history[0].content)
        # Ensure the third message is the user's follow-up question
        self.assertIn(q2, history[2].content)
        # Print the chat history for debugging
        print("Chat history:", [(type(m).__name__, m.content) for m in history])

if __name__ == "__main__":
    unittest.main()
