"""Memory manager for conversation history with vector search.

Manages conversation history using FAISS vector database for semantic search.
Combines recent conversation turns with semantically similar past turns to
provide relevant context to the LLM.

Key Features:
- Vector embeddings for semantic similarity search
- Persistent storage (survives restarts)
- Hybrid context retrieval (recent + similar turns)
- Token limit management

💾 PERFORMANCE IMPACT:
- max_turns: More turns = better context but slower search
- similarity_top_k: More results = better context but higher token usage
- recent_turns: Always included for temporal context
"""

import asyncio
import logging
import os
from typing import List, Optional
import time
import pickle

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.models import ConversationTurn, Message


logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages conversation history with semantic search using FAISS."""
    
    def __init__(
        self,
        openai_api_key: str,
        max_turns: int = 10,
        similarity_top_k: int = 5,
        recent_turns: int = 3,
        vector_db_path: str = "./data/conversation_memory"
    ):
        """
        Initialize memory manager.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            max_turns: Maximum conversation turns to store (default: 10)
                💾 PERFORMANCE: More turns = better long-term memory but slower
            similarity_top_k: Number of similar turns to retrieve (default: 5)
                💾 PERFORMANCE: More results = better context but higher token usage
            recent_turns: Number of recent turns to always include (default: 3)
                💾 PERFORMANCE: Always included for temporal context
            vector_db_path: Path to persist FAISS index (default: ./data/conversation_memory)
        """
        self.max_turns = max_turns
        self.similarity_top_k = similarity_top_k
        self.recent_turns = recent_turns
        self.vector_db_path = vector_db_path
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Conversation storage
        self._turns: List[ConversationTurn] = []
        self._vector_store: Optional[FAISS] = None
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
        
        # Try to load existing vector store
        self._load_vector_store()
        
        logger.info(
            f"MemoryManager initialized: max_turns={max_turns}, "
            f"similarity_top_k={similarity_top_k}, recent_turns={recent_turns}"
        )
    
    def _load_vector_store(self) -> None:
        """Load FAISS vector store from disk if it exists."""
        index_path = f"{self.vector_db_path}.faiss"
        pkl_path = f"{self.vector_db_path}.pkl"
        
        if os.path.exists(index_path) and os.path.exists(pkl_path):
            try:
                # Load FAISS index
                self._vector_store = FAISS.load_local(
                    self.vector_db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Load conversation turns
                with open(pkl_path, 'rb') as f:
                    self._turns = pickle.load(f)
                
                logger.info(
                    f"Loaded {len(self._turns)} conversation turns from {self.vector_db_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")
                self._vector_store = None
                self._turns = []
        else:
            logger.info("No existing vector store found, starting fresh")
    
    def _save_vector_store(self) -> None:
        """Save FAISS vector store to disk."""
        try:
            if self._vector_store:
                # Save FAISS index
                self._vector_store.save_local(self.vector_db_path)
                
                # Save conversation turns
                pkl_path = f"{self.vector_db_path}.pkl"
                with open(pkl_path, 'wb') as f:
                    pickle.dump(self._turns, f)
                
                logger.debug(f"Saved vector store to {self.vector_db_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    async def add_turn(
        self, 
        user_input: str, 
        assistant_response: str
    ) -> None:
        """
        Store a conversation turn with embedding.
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
        """
        timestamp = time.time()
        
        # Create conversation turn
        turn = ConversationTurn(
            user_input=user_input,
            assistant_response=assistant_response,
            timestamp=timestamp
        )
        
        # Generate embedding for the turn
        # Combine user input and response for better semantic search
        turn_text = f"User: {user_input}\nAssistant: {assistant_response}"
        
        try:
            # Get embedding
            embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                turn_text
            )
            turn.embedding = embedding
            
            # Add to turns list
            self._turns.append(turn)
            
            # Limit turns to max_turns
            if len(self._turns) > self.max_turns:
                removed_turn = self._turns.pop(0)
                logger.debug(f"Removed oldest turn (timestamp: {removed_turn.timestamp})")
            
            # Create or update vector store
            doc = Document(
                page_content=turn_text,
                metadata={
                    "timestamp": timestamp,
                    "user_input": user_input,
                    "assistant_response": assistant_response
                }
            )
            
            if self._vector_store is None:
                # Create new vector store
                self._vector_store = await asyncio.to_thread(
                    FAISS.from_documents,
                    [doc],
                    self.embeddings
                )
            else:
                # Add to existing vector store
                await asyncio.to_thread(
                    self._vector_store.add_documents,
                    [doc]
                )
            
            logger.info(f"Added conversation turn (total: {len(self._turns)})")
            
            # Save to disk
            self._save_vector_store()
            
        except Exception as e:
            logger.error(f"Error adding conversation turn: {e}", exc_info=True)
    
    async def get_context(
        self, 
        current_input: str,
        max_tokens: int = 2000
    ) -> List[Message]:
        """
        Retrieve relevant conversation context.
        
        Uses hybrid strategy:
        1. Similarity search for semantically relevant turns
        2. Recent turns for temporal context
        
        Args:
            current_input: Current user input for similarity search
            max_tokens: Maximum tokens to include in context (approximate)
            
        Returns:
            List of Message objects for LLM context
        """
        if not self._turns:
            logger.debug("No conversation history available")
            return []
        
        try:
            context_turns: List[ConversationTurn] = []
            
            # Get recent turns (always include)
            recent = self._turns[-self.recent_turns:]
            context_turns.extend(recent)
            
            # Get similar turns (if vector store exists)
            if self._vector_store and len(self._turns) > self.recent_turns:
                try:
                    # Search for similar turns
                    similar_docs = await asyncio.to_thread(
                        self._vector_store.similarity_search,
                        current_input,
                        k=self.similarity_top_k
                    )
                    
                    # Extract turns from similar documents
                    for doc in similar_docs:
                        # Find matching turn by timestamp
                        timestamp = doc.metadata.get("timestamp")
                        matching_turn = next(
                            (t for t in self._turns if t.timestamp == timestamp),
                            None
                        )
                        
                        if matching_turn and matching_turn not in context_turns:
                            context_turns.append(matching_turn)
                    
                    logger.debug(
                        f"Retrieved {len(similar_docs)} similar turns, "
                        f"total context: {len(context_turns)} turns"
                    )
                    
                except Exception as e:
                    logger.warning(f"Error in similarity search: {e}")
            
            # Sort by timestamp to maintain chronological order
            context_turns.sort(key=lambda t: t.timestamp)
            
            # Convert to Message format and limit tokens
            messages: List[Message] = []
            total_chars = 0
            max_chars = max_tokens * 4  # Rough approximation: 1 token ≈ 4 chars
            
            for turn in context_turns:
                turn_chars = len(turn.user_input) + len(turn.assistant_response)
                
                if total_chars + turn_chars > max_chars:
                    logger.debug(f"Context limit reached, truncating at {len(messages)} messages")
                    break
                
                # Add user message
                messages.append(Message(
                    role="user",
                    content=turn.user_input,
                    timestamp=turn.timestamp
                ))
                
                # Add assistant message
                messages.append(Message(
                    role="assistant",
                    content=turn.assistant_response,
                    timestamp=turn.timestamp
                ))
                
                total_chars += turn_chars
            
            logger.info(f"Retrieved {len(messages)} context messages")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return []
    
    def clear(self) -> None:
        """Clear all conversation history."""
        logger.info("Clearing conversation history...")
        
        self._turns.clear()
        self._vector_store = None
        
        # Remove persisted files
        try:
            index_path = f"{self.vector_db_path}.faiss"
            pkl_path = f"{self.vector_db_path}.pkl"
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
            
            logger.info("Conversation history cleared")
        except Exception as e:
            logger.error(f"Error clearing persisted data: {e}")
    
    def get_turn_count(self) -> int:
        """Get number of stored conversation turns."""
        return len(self._turns)
    
    def get_all_turns(self) -> List[ConversationTurn]:
        """Get all stored conversation turns."""
        return self._turns.copy()
