"""
Multi-Agent System Example

This example demonstrates how to create and orchestrate multiple specialized agents
working together to complete complex tasks.
"""

import asyncio
from dotenv import load_dotenv

from src.adk_alpha.agents import (
    create_multi_agent_system,
    ResearchAgent,
    AnalystAgent,
    WriterAgent,
    DEFAULT_RESEARCH_WORKFLOW,
    PARALLEL_ANALYSIS_TASKS,
)
from src.adk_alpha.config import setup_logging, load_config


async def sequential_workflow_example():
    """Demonstrate sequential multi-agent workflow."""
    print("🔄 Sequential Multi-Agent Workflow Example")
    print("=" * 50)
    
    # Create specialized agents
    agent_configs = [
        {
            "type": "research",
            "agent_name": "researcher",
            "instructions": (
                "You are a research specialist. Gather comprehensive information "
                "on the given topic. Focus on factual, reliable sources."
            )
        },
        {
            "type": "analyst", 
            "agent_name": "analyst",
            "instructions": (
                "You are a data analyst. Process and analyze the research findings. "
                "Identify key patterns, trends, and insights."
            )
        },
        {
            "type": "writer",
            "agent_name": "writer",
            "instructions": (
                "You are a professional writer. Create clear, well-structured "
                "reports based on research and analysis."
            )
        }
    ]
    
    # Create the multi-agent system
    orchestrator = create_multi_agent_system(agent_configs)
    
    # Define the workflow for researching AI trends
    workflow = [
        {
            "agent_name": "researcher",
            "query": "Research the latest trends in artificial intelligence for 2024"
        },
        {
            "agent_name": "analyst",
            "query": "Analyze the research findings and identify the most significant AI trends"
        },
        {
            "agent_name": "writer",
            "query": "Write a comprehensive summary report on the AI trends analysis"
        }
    ]
    
    print("🚀 Starting sequential workflow...")
    print("Topic: Latest AI Trends 2024")
    print()
    
    try:
        result = await orchestrator.execute_sequential(workflow)
        
        if result["status"] == "success":
            print("✅ Workflow completed successfully!")
            print(f"⏱️  Total execution time: {result['execution_time']:.2f}s")
            print(f"📊 Steps completed: {len(result['results'])}")
            print()
            
            # Display results from each step
            for step_result in result['results']:
                step_num = step_result['step']
                agent_name = step_result['agent_name']
                status = step_result['result']['status']
                
                print(f"Step {step_num} - {agent_name.title()}: {status}")
                
                if status == "success":
                    content = step_result['result']['result']['content']
                    # Show truncated content
                    print(f"   📝 Output: {content[:150]}...")
                    if len(content) > 150:
                        print("       (truncated)")
                else:
                    error = step_result['result']['error']
                    print(f"   ❌ Error: {error}")
                print()
        else:
            print(f"❌ Workflow failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Exception during workflow: {str(e)}")


async def parallel_workflow_example():
    """Demonstrate parallel multi-agent execution."""
    print("⚡ Parallel Multi-Agent Workflow Example")
    print("=" * 50)
    
    # Create agents for parallel execution
    agent_configs = [
        {
            "type": "research",
            "agent_name": "tech_researcher",
            "instructions": "Research technology trends and innovations."
        },
        {
            "type": "research", 
            "agent_name": "market_researcher",
            "instructions": "Research market trends and business insights."
        },
        {
            "type": "analyst",
            "agent_name": "data_analyst",
            "instructions": "Analyze quantitative data and metrics."
        }
    ]
    
    orchestrator = create_multi_agent_system(agent_configs)
    
    # Define parallel tasks
    parallel_tasks = [
        {
            "agent_name": "tech_researcher",
            "query": "Research emerging technologies in 2024"
        },
        {
            "agent_name": "market_researcher", 
            "query": "Research market opportunities in the tech sector"
        },
        {
            "agent_name": "data_analyst",
            "query": "Analyze growth metrics for the technology industry"
        }
    ]
    
    print("🚀 Starting parallel execution...")
    print("Topic: Technology Sector Analysis")
    print()
    
    try:
        result = await orchestrator.execute_parallel(parallel_tasks)
        
        if result["status"] == "success":
            print("✅ Parallel execution completed!")
            print(f"⏱️  Total execution time: {result['execution_time']:.2f}s")
            print(f"📊 Tasks completed: {result['successful_tasks']}/{result['total_tasks']}")
            print()
            
            # Display results from each task
            for task_result in result['results']:
                task_num = task_result['task']
                agent_name = task_result['agent_name']
                status = task_result['result']['status']
                
                print(f"Task {task_num} - {agent_name}: {status}")
                
                if status == "success":
                    content = task_result['result']['result']['content']
                    exec_time = task_result['result']['execution_time']
                    print(f"   📝 Output: {content[:100]}...")
                    print(f"   ⏱️  Time: {exec_time:.2f}s")
                else:
                    error = task_result['result']['error']
                    print(f"   ❌ Error: {error}")
                print()
        else:
            print(f"❌ Parallel execution failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Exception during parallel execution: {str(e)}")


async def custom_workflow_example():
    """Demonstrate custom workflow patterns."""
    print("🎯 Custom Workflow Example")
    print("=" * 50)
    
    # Create a mixed workflow combining sequential and parallel patterns
    agent_configs = [
        {
            "type": "research",
            "agent_name": "primary_researcher"
        },
        {
            "type": "research",
            "agent_name": "secondary_researcher"  
        },
        {
            "type": "analyst",
            "agent_name": "data_analyst"
        },
        {
            "type": "writer",
            "agent_name": "report_writer"
        }
    ]
    
    orchestrator = create_multi_agent_system(agent_configs)
    
    print("🚀 Starting custom workflow...")
    print("Pattern: Parallel Research → Analysis → Report")
    print()
    
    try:
        # Step 1: Parallel research
        research_tasks = [
            {
                "agent_name": "primary_researcher",
                "query": "Research current state of renewable energy technology"
            },
            {
                "agent_name": "secondary_researcher",
                "query": "Research renewable energy market trends and policies"
            }
        ]
        
        print("📊 Phase 1: Parallel Research")
        research_result = await orchestrator.execute_parallel(research_tasks)
        
        if research_result["status"] != "success":
            print(f"❌ Research phase failed: {research_result['error']}")
            return
        
        print(f"✅ Research completed ({research_result['execution_time']:.2f}s)")
        
        # Combine research results
        combined_research = ""
        for task_result in research_result['results']:
            if task_result['result']['status'] == 'success':
                combined_research += task_result['result']['result']['content'] + "\n\n"
        
        # Step 2: Sequential analysis and writing
        analysis_workflow = [
            {
                "agent_name": "data_analyst",
                "query": f"Analyze this renewable energy research data: {combined_research[:1000]}..."
            },
            {
                "agent_name": "report_writer",
                "query": "Create a comprehensive report on renewable energy based on the analysis"
            }
        ]
        
        print("\n📈 Phase 2: Analysis and Report Generation")
        analysis_result = await orchestrator.execute_sequential(analysis_workflow)
        
        if analysis_result["status"] == "success":
            print(f"✅ Analysis and reporting completed ({analysis_result['execution_time']:.2f}s)")
            
            # Show final report
            final_step = analysis_result['results'][-1]
            if final_step['result']['status'] == 'success':
                final_report = final_step['result']['result']['content']
                print("\n📄 Final Report Preview:")
                print("-" * 30)
                print(final_report[:300] + "..." if len(final_report) > 300 else final_report)
        else:
            print(f"❌ Analysis phase failed: {analysis_result['error']}")
    
    except Exception as e:
        print(f"❌ Exception during custom workflow: {str(e)}")


async def agent_comparison_example():
    """Compare different agent types on the same task."""
    print("🔍 Agent Comparison Example")
    print("=" * 50)
    
    # Create different types of agents
    research_agent = ResearchAgent(agent_name="specialist_researcher")
    analyst_agent = AnalystAgent(agent_name="specialist_analyst") 
    writer_agent = WriterAgent(agent_name="specialist_writer")
    
    # Test the same query with different agents
    test_query = "Explain the concept of machine learning in simple terms"
    
    agents = [
        ("Research Agent", research_agent),
        ("Analyst Agent", analyst_agent),
        ("Writer Agent", writer_agent)
    ]
    
    print("🧪 Testing query with different agent types:")
    print(f"Query: {test_query}")
    print()
    
    for agent_type, agent in agents:
        try:
            print(f"🤖 {agent_type}:")
            result = await agent.execute(test_query)
            
            if result["status"] == "success":
                response = result['result']['content']
                exec_time = result['execution_time']
                print(f"   ✅ Response ({exec_time:.2f}s): {response[:200]}...")
                if len(response) > 200:
                    print("      (truncated)")
            else:
                print(f"   ❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
        
        print()


async def main():
    """Main function to run all examples."""
    # Load environment
    load_dotenv()
    setup_logging("INFO")
    
    print("ADK Alpha - Multi-Agent System Examples")
    print("=" * 60)
    print()
    
    examples = [
        ("Sequential Workflow", sequential_workflow_example),
        ("Parallel Workflow", parallel_workflow_example),
        ("Custom Workflow", custom_workflow_example),
        ("Agent Comparison", agent_comparison_example),
    ]
    
    for example_name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"❌ Error in {example_name}: {str(e)}")
        
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
