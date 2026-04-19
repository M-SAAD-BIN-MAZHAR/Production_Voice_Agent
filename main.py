"""Main entry point for the real-time voice agent system."""

import asyncio
import logging
import sys
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_manager import ConfigManager
from src.voice_agent import VoiceAgent


# Global agent instance for signal handling
agent_instance = None


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {sig}, initiating shutdown...")
    
    if agent_instance:
        # Create a new event loop for shutdown if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule shutdown in the running loop
                asyncio.create_task(agent_instance.stop())
            else:
                # Run shutdown in new loop
                asyncio.run(agent_instance.stop())
        except Exception as e:
            logger.error(f"Error during signal handler shutdown: {e}")
    
    sys.exit(0)


async def main():
    """Main async entry point."""
    global agent_instance
    
    # Load configuration
    try:
        config = ConfigManager.load()
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("\n📋 Please ensure you have:")
        print("   1. Created config/config.yaml from config/config.example.yaml")
        print("   2. Or set environment variables (see config/.env.example)")
        print("   3. Provided required API keys (OPENAI_API_KEY, DEEPGRAM_API_KEY)")
        return 1
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🎤 Real-Time Voice Agent System starting...")
    
    if config.voice_input_mode == "push_to_talk":
        if config.push_to_talk_stdin_arm_seconds > 0:
            logger.info(
                "Push-to-talk: hold '%s' while speaking, or press Enter in this terminal "
                "to arm the mic for ~%.0fs (helps when Space does not work in an IDE terminal).",
                config.push_to_talk_key,
                config.push_to_talk_stdin_arm_seconds,
            )
        else:
            logger.info(
                "Push-to-talk: hold key '%s' while speaking so mic audio reaches STT/VAD.",
                config.push_to_talk_key,
            )
    elif config.voice_input_mode == "wake_word":
        logger.info(
            "Wake-word mode: say a trained phrase first; window stays open %.0fs.",
            config.wake_word_window_seconds,
        )
    
    # Create voice agent
    agent_instance = VoiceAgent(config)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start voice agent
        await agent_instance.start()
        logger.info("✅ Voice agent stopped")
        return 0
        
    except KeyboardInterrupt:
        logger.info("⌨️  Keyboard interrupt received")
        await agent_instance.stop()
        return 0
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        await agent_instance.stop()
        return 1


if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console FIRST
    import sys
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    # Now safe to print emojis
    print("🎤 Real-Time Voice Agent System")
    print("=" * 50)
    print("Press Ctrl+C to stop\n")
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
