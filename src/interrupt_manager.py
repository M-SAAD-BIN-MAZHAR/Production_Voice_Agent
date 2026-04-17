"""Interrupt manager for handling barge-in during agent responses."""

import asyncio
import logging
import time
from typing import Optional, Callable, Awaitable

from src.models import VADEvent, VADEventType, SystemState


logger = logging.getLogger(__name__)


class InterruptManager:
    """Manages barge-in interrupts when user speaks during agent response."""
    
    def __init__(self):
        """Initialize interrupt manager."""
        self._state = SystemState.LISTENING
        self._monitoring = False
        self._interrupt_callbacks: list[Callable[[], Awaitable[None]]] = []
        
        # Timing tracking
        self._last_interrupt_time: Optional[float] = None
        self._interrupt_count = 0
        self._interrupt_debounce_ms = 3500  # Minimum 3.5 seconds between interrupts to prevent cutting off responses
        
        logger.info("InterruptManager initialized")
    
    def set_state(self, state: SystemState) -> None:
        """
        Update system state.
        
        Args:
            state: New system state
        """
        old_state = self._state
        self._state = state
        
        if old_state != state:
            logger.debug(f"State transition: {old_state.value} → {state.value}")
    
    def get_state(self) -> SystemState:
        """Get current system state."""
        return self._state
    
    def register_interrupt_callback(
        self, 
        callback: Callable[[], Awaitable[None]]
    ) -> None:
        """
        Register callback to be called on interrupt.
        
        Args:
            callback: Async function to call when interrupt occurs
        """
        self._interrupt_callbacks.append(callback)
        logger.debug(f"Registered interrupt callback: {callback.__name__}")
    
    async def monitor_interrupts(
        self,
        vad_event_queue: asyncio.Queue[VADEvent],
        is_playing_callback: Callable[[], bool]
    ) -> None:
        """
        Monitor for barge-in conditions.
        
        Detects when user starts speaking while agent is speaking.
        
        Args:
            vad_event_queue: Queue of VAD events to monitor
            is_playing_callback: Function that returns True if audio is playing
        """
        self._monitoring = True
        logger.info("Started monitoring for interrupts")
        
        try:
            while self._monitoring:
                # Get next VAD event
                vad_event = await vad_event_queue.get()
                
                # Check for barge-in condition:
                # User starts speaking (SPEECH_START) while agent is speaking
                if (vad_event.event_type == VADEventType.SPEECH_START and 
                    is_playing_callback()):
                    
                    # Check debounce - prevent rapid successive interrupts
                    current_time = time.time()
                    if self._last_interrupt_time is not None:
                        time_since_last_interrupt_ms = (current_time - self._last_interrupt_time) * 1000
                        if time_since_last_interrupt_ms < self._interrupt_debounce_ms:
                            logger.debug(
                                f"Interrupt debounced: {time_since_last_interrupt_ms:.0f}ms "
                                f"< {self._interrupt_debounce_ms}ms"
                            )
                            vad_event_queue.task_done()
                            continue
                    
                    # Calculate detection latency
                    detection_time = time.time()
                    detection_latency_ms = (detection_time - vad_event.timestamp) * 1000
                    
                    logger.info(
                        f"BARGE-IN detected! Latency: {detection_latency_ms:.1f}ms, "
                        f"confidence: {vad_event.confidence:.2f}"
                    )
                    
                    # Log warning if detection latency exceeds threshold
                    if detection_latency_ms > 100:
                        logger.warning(
                            f"Barge-in detection latency: {detection_latency_ms:.1f}ms (> 100ms)"
                        )
                    
                    # Handle the interrupt
                    await self.handle_interrupt()
                
                # Mark task as done
                vad_event_queue.task_done()
                
        except asyncio.CancelledError:
            logger.info("Interrupt monitoring cancelled")
            raise
        except Exception as e:
            logger.error(f"Error monitoring interrupts: {e}", exc_info=True)
        finally:
            self._monitoring = False
    
    async def handle_interrupt(self) -> None:
        """
        Execute interrupt sequence.
        
        This method:
        1. Updates system state to INTERRUPTED
        2. Calls all registered interrupt callbacks
        3. Logs interrupt event
        """
        interrupt_start_time = time.time()
        
        # Update state
        self.set_state(SystemState.INTERRUPTED)
        
        # Track interrupt
        self._last_interrupt_time = interrupt_start_time
        self._interrupt_count += 1
        
        logger.info(
            f"Handling interrupt #{self._interrupt_count} at {interrupt_start_time:.3f}"
        )
        
        # Execute all interrupt callbacks
        callback_tasks = []
        for callback in self._interrupt_callbacks:
            try:
                task = asyncio.create_task(callback())
                callback_tasks.append(task)
            except Exception as e:
                logger.error(f"Error creating interrupt callback task: {e}")
        
        # Wait for all callbacks to complete
        if callback_tasks:
            try:
                await asyncio.gather(*callback_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error executing interrupt callbacks: {e}")
        
        # Calculate interrupt handling time
        interrupt_duration_ms = (time.time() - interrupt_start_time) * 1000
        logger.info(f"Interrupt handled in {interrupt_duration_ms:.1f}ms")
        
        # Transition back to LISTENING state
        self.set_state(SystemState.LISTENING)
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring for interrupts."""
        logger.info("Stopping interrupt monitoring...")
        self._monitoring = False
    
    def is_monitoring(self) -> bool:
        """Check if currently monitoring for interrupts."""
        return self._monitoring
    
    def get_interrupt_stats(self) -> dict:
        """
        Get interrupt statistics.
        
        Returns:
            Dictionary with interrupt statistics
        """
        return {
            "total_interrupts": self._interrupt_count,
            "last_interrupt_time": self._last_interrupt_time,
            "current_state": self._state.value
        }
