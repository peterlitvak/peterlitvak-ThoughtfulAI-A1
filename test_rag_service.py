import os
import unittest
from rag_backend import RAGService, InMemoryChatHistoryProvider
from dotenv import load_dotenv

class TestRAGServiceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()
        cls.api_key = os.environ.get("OPENAI_API_KEY")
        assert cls.api_key, "OPENAI_API_KEY must be set in the environment for integration test."
        cls.kb_path = "data.json"  # Use your real data.json
        cls.system_prompt = "You are a helpful AI."
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
        answer = self.rag.chat("What is Python?", session_id="integration1", stream=False)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        print("Non-streaming answer:", answer)

    def test_chat_streaming(self):
        chunks = list(self.rag.chat("What is AI?", session_id="integration2", stream=True))
        self.assertTrue(any(isinstance(c, str) and c for c in chunks))
        print("Streaming answer chunks:", chunks)

    def test_knowledge_base_usage(self):
        # This question/answer pair must exist in your data.json
        question = "What does the eligibility verification agent (EVA) do?"
        expected_phrase = "EVA automates the process of verifying a patient’s eligibility and benefits information"
        answer = self.rag.chat(question, session_id="kb_usage", stream=False)
        self.assertIn(expected_phrase, answer)
        print("Knowledge base answer:", answer)

    def test_chat_history(self):
        session_id = "history_test"
        # First turn
        q1 = "What is EVA?"
        a1 = self.rag.chat(q1, session_id=session_id, stream=False)
        # Second turn, context should include previous Q/A
        q2 = "What does it do?"
        a2 = self.rag.chat(q2, session_id=session_id, stream=False)
        # Ensure history is tracked and both user and AI messages are present
        history = self.rag.chat_memory_provider.get_history(session_id)
        self.assertGreaterEqual(len(history), 4)  # 2 user + 2 AI
        self.assertIn(q1, history[0].content)
        self.assertIn(q2, history[2].content)
        print("Chat history:", [(type(m).__name__, m.content) for m in history])

if __name__ == "__main__":
    unittest.main()
