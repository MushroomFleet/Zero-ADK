"""
Basic Agent Example

This example demonstrates how to create and use a basic agent
with the ADK Alpha implementation.
"""

import asyncio
import os
from dotenv import load_dotenv

from src.adk_alpha.agents import BasicAgent, create_basic_agent
from src.adk_alpha.config import setup_logging, validate_config, load_config


async def main():
    """Main example function."""
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    setup_logging("INFO")
    
    # Load and validate configuration
    config = load_config()
    
    try:
        validate_config(config)
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please check your .env file and ensure required variables are set.")
        return
    
    # Create a basic agent
    agent = create_basic_agent(
        agent_name="example_assistant",
        instructions=(
            "You are a helpful assistant. You can perform calculations, "
            "analyze text, and answer questions. Be concise and accurate."
        ),
        enable_search=False,  # Disable for this example
        enable_calculator=True,
        enable_text_processing=True
    )
    
    print(f"🤖 Created agent: {agent.agent_name}")
    
    # Example queries to test the agent
    test_queries = [
        "Calculate 15 * 23 + 7",
        "Analyze this text and count words: 'The quick brown fox jumps over the lazy dog'",
        "What is the square root of 144?",
        "Count characters in 'Hello, World!'"
    ]
    
    print("\n🧪 Testing agent with example queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        try:
            result = await agent.execute(query)
            
            if result["status"] == "success":
                print(f"✅ Response: {result['result']['content'][:200]}...")
                if len(result['result']['content']) > 200:
                    print("   (truncated)")
                print(f"   Execution time: {result['execution_time']:.2f}s")
                
                # Show tool calls if any
                if result['result']['tool_calls']:
                    print(f"   🔧 Tools used: {len(result['result']['tool_calls'])}")
            else:
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
        
        print("-" * 50)


async def interactive_mode():
    """Interactive mode for testing the agent."""
    load_dotenv()
    setup_logging("INFO")
    
    agent = create_basic_agent(
        agent_name="interactive_assistant",
        instructions=(
            "You are an interactive assistant. Help the user with calculations, "
            "text analysis, and general questions. Be helpful and conversational."
        ),
        enable_search=False,
        enable_calculator=True,
        enable_text_processing=True
    )
    
    print("🤖 Interactive Agent Ready!")
    print("Type 'quit' or 'exit' to stop, 'help' for available commands.")
    print("-" * 50)
    
    session_id = "interactive_session"
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("""
Available commands:
- Ask any question or request
- 'calc [expression]' - Perform calculations
- 'analyze [text]' - Analyze text
- 'quit' or 'exit' - Exit the program
                """)
                continue
            
            if not query:
                continue
            
            print("🤖 Assistant: ", end="", flush=True)
            
            result = await agent.execute(query, session_id=session_id)
            
            if result["status"] == "success":
                print(result['result']['content'])
                
                # Show execution info
                if result.get('execution_time', 0) > 1:
                    print(f"   ⏱️  Execution time: {result['execution_time']:.2f}s")
            else:
                print(f"❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    print("ADK Alpha - Basic Agent Example")
    print("=" * 40)
    
    # Check if we should run in interactive mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_mode())
    else:
        print("Running example queries...")
        print("(Use 'python basic_agent_example.py interactive' for interactive mode)")
        print()
        asyncio.run(main())
