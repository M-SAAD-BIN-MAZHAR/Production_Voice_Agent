"""Simple terminal UI for voice agent."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from src.models import SystemState


logger = logging.getLogger(__name__)


class TerminalUI:
    """Simple terminal-based UI with basic print output."""
    
    def __init__(self):
        """Initialize terminal UI."""
        self._running = False
        self._current_state = SystemState.INITIALIZING
        self._error_message: Optional[str] = None
        
        logger.info("TerminalUI initialized")
    
    async def start(self) -> None:
        """Start the UI."""
        if self._running:
            logger.warning("TerminalUI already running")
            return
        
        self._running = True
        logger.info("TerminalUI started")
        print("\n" + "="*60)
        print("🎤 Real-Time Voice Agent Started")
        print("="*60 + "\n")
    
    async def update_transcript(
        self,
        role: str,
        content: str,
        is_partial: bool = False
    ) -> None:
        """
        Update transcript display.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            is_partial: Whether this is a partial update
        """
        if not self._running:
            return
        
        # Format based on role
        if role == "user":
            prefix = "👤 You"
        else:
            prefix = "🤖 Agent"
        
        # Show partial updates with ellipsis
        if is_partial:
            print(f"{prefix}: {content}...", end="\r", flush=True)
        else:
            print(f"{prefix}: {content}")
    
    async def update_status(self, state: SystemState) -> None:
        """
        Update system state display.
        
        Args:
            state: Current system state
        """
        if not self._running:
            return
        
        self._current_state = state
        
        # Map state to emoji and message
        state_map = {
            SystemState.INITIALIZING: ("⏳", "Initializing..."),
            SystemState.LISTENING: ("👂", "Listening..."),
            SystemState.PROCESSING: ("🧠", "Processing..."),
            SystemState.SPEAKING: ("🔊", "Speaking..."),
            SystemState.ERROR: ("❌", "Error"),
        }
        
        emoji, message = state_map.get(state, ("❓", "Unknown"))
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"[{timestamp}] {emoji} {message}")
    
    async def display_error(self, message: str) -> None:
        """
        Display error message.
        
        Args:
            message: Error message to display
        """
        if not self._running:
            return
        
        self._error_message = message
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ❌ ERROR: {message}")
    
    async def clear_error(self) -> None:
        """Clear error message."""
        self._error_message = None
    
    async def stop(self) -> None:
        """Stop the UI."""
        if not self._running:
            return
        
        self._running = False
        print("\n" + "="*60)
        print("👋 Voice Agent Stopped")
        print("="*60 + "\n")
        logger.info("TerminalUI stopped")
    
    def is_running(self) -> bool:
        """Check if UI is running."""
        return self._running
