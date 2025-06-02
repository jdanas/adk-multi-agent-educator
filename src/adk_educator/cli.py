"""
Command Line Interface for the ADK Multi-Agent Educator system.

This module provides an interactive CLI that allows users to:
- Start educational sessions
- Ask questions to specific agents
- Get multi-agent responses
- View session summaries and agent capabilities
"""

import asyncio
import sys
from typing import Optional, List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

from .core.session_manager import SessionManager
from .core.coordinator import AgentCoordinator
from .agents.math_agent import MathAgent
from .agents.science_agent import ScienceAgent
from .agents.music_agent import MusicAgent
from .config import StudentRequest, SubjectType, DifficultyLevel, SystemConfig

app = typer.Typer(
    name="adk-educator",
    help="ADK Multi-Agent Educator - Interactive learning with specialized AI agents",
    rich_markup_mode="rich"
)

console = Console()


class EducatorCLI:
    """Main CLI application class."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.session_manager = SessionManager()
        self.coordinator = AgentCoordinator(self.session_manager)
        self.current_session_id: Optional[str] = None
        self.current_student_id: Optional[str] = None
        self._setup_agents()
    
    def _setup_agents(self):
        """Initialize and register all agents."""
        try:
            # Create and register agents
            math_agent = MathAgent()
            science_agent = ScienceAgent()
            music_agent = MusicAgent()
            
            self.session_manager.register_agent(math_agent)
            self.session_manager.register_agent(science_agent)
            self.session_manager.register_agent(music_agent)
            
            console.print("✅ All agents initialized successfully!", style="green")
            
        except Exception as e:
            console.print(f"❌ Error initializing agents: {str(e)}", style="red")
            console.print("💡 Make sure your Google API key is set in the .env file", style="yellow")
    
    async def start_session(self, student_name: str) -> str:
        """Start a new learning session."""
        try:
            session_id = self.session_manager.create_session(student_name)
            self.current_session_id = session_id
            self.current_student_id = student_name
            
            console.print(Panel(
                f"🎓 Welcome to ADK Multi-Agent Educator, {student_name}!\n\n"
                f"Session ID: {session_id}\n\n"
                "You can ask questions about:\n"
                "📐 Mathematics (algebra, geometry, calculus, statistics)\n"
                "🔬 Science (physics, chemistry, biology, earth science)\n"
                "🎵 Music (theory, history, composition, performance)\n\n"
                "Type 'help' for commands or start asking questions!",
                title="🌟 New Learning Session Started",
                border_style="blue"
            ))
            
            return session_id
            
        except Exception as e:
            console.print(f"❌ Error starting session: {str(e)}", style="red")
            raise
    
    async def ask_question(
        self, 
        question: str, 
        subject: Optional[SubjectType] = None,
        topic: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None
    ):
        """Ask a question to the agents."""
        if not self.current_session_id:
            console.print("❌ No active session. Please start a session first.", style="red")
            return
        
        try:
            console.print(f"🤔 Processing your question: [italic]{question}[/italic]")
            
            # Auto-detect subject if not specified
            if not subject:
                relevant_subjects = self.coordinator.identify_relevant_subjects(question)
                if len(relevant_subjects) == 1:
                    subject = relevant_subjects[0]
                elif len(relevant_subjects) > 1:
                    # Multi-subject question
                    await self._handle_multi_subject_question(
                        question, relevant_subjects, topic, difficulty
                    )
                    return
                else:
                    subject = SubjectType.MATH  # Default fallback
            
            # Create the request
            request = StudentRequest(
                session_id=self.current_session_id,
                student_id=self.current_student_id,
                subject=subject,
                topic=topic or "general",
                difficulty=difficulty,
                specific_question=question
            )
            
            # Get response from the session manager
            with console.status("🧠 Thinking..."):
                response = await self.session_manager.process_student_request(request)
            
            # Display the response
            self._display_response(response)
            
        except Exception as e:
            console.print(f"❌ Error processing question: {str(e)}", style="red")
    
    async def _handle_multi_subject_question(
        self,
        question: str,
        subjects: List[SubjectType],
        topic: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None
    ):
        """Handle questions that involve multiple subjects."""
        console.print(f"🌐 This question involves multiple subjects: {', '.join([s.value for s in subjects])}")
        console.print("Getting coordinated response from multiple experts...")
        
        request = StudentRequest(
            session_id=self.current_session_id,
            student_id=self.current_student_id,
            subject=subjects[0],  # Primary subject
            topic=topic or "interdisciplinary",
            difficulty=difficulty,
            specific_question=question
        )
        
        with console.status("🤝 Coordinating with multiple experts..."):
            response = await self.coordinator.process_complex_query(request, subjects)
        
        self._display_response(response)
    
    def _display_response(self, response):
        """Display an agent response in a formatted way."""
        # Main response
        console.print(Panel(
            Markdown(response.response_text),
            title=f"💡 {response.agent_name} ({response.subject.value.title()})",
            border_style="green" if response.confidence_score > 0.7 else "yellow"
        ))
        
        # Confidence indicator
        confidence_color = "green" if response.confidence_score > 0.7 else "yellow" if response.confidence_score > 0.4 else "red"
        console.print(f"🎯 Confidence: {response.confidence_score:.1%}", style=confidence_color)
        
        # Follow-up questions
        if response.suggested_follow_ups:
            console.print("\n💭 Suggested follow-up questions:")
            for i, follow_up in enumerate(response.suggested_follow_ups, 1):
                console.print(f"  {i}. {follow_up}")
        
        # Resources
        if response.resources:
            console.print("\n📚 Helpful resources:")
            for resource in response.resources:
                console.print(f"  • {resource}")
        
        console.print()  # Add spacing
    
    def show_agents(self):
        """Display information about available agents."""
        agents_info = self.session_manager.get_available_agents()
        
        table = Table(title="🤖 Available Educational Agents")
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Subject", style="magenta")
        table.add_column("Specializations", style="green")
        table.add_column("Difficulty Levels", style="yellow")
        
        for agent_name, info in agents_info.items():
            specializations = ", ".join(info["specializations"][:3])
            if len(info["specializations"]) > 3:
                specializations += f" (+{len(info['specializations']) - 3} more)"
            
            difficulty_levels = ", ".join(info["difficulty_levels"])
            
            table.add_row(
                agent_name,
                info["subject"],
                specializations,
                difficulty_levels
            )
        
        console.print(table)
    
    def show_session_summary(self):
        """Display current session summary."""
        if not self.current_session_id:
            console.print("❌ No active session", style="red")
            return
        
        summary = self.session_manager.get_session_summary(self.current_session_id)
        
        console.print(Panel(
            f"📊 **Session Summary**\n\n"
            f"Student: {summary['student_id']}\n"
            f"Duration: {summary['duration_minutes']} minutes\n"
            f"Total Interactions: {summary['total_interactions']}\n"
            f"Active Subjects: {', '.join(summary['active_subjects'])}\n"
            f"Preferred Agents: {', '.join(summary['preferred_agents']) if summary['preferred_agents'] else 'None yet'}",
            title="📈 Learning Progress",
            border_style="blue"
        ))


# Create global CLI instance
cli = EducatorCLI()


@app.command()
def start(
    student_name: str = typer.Argument(..., help="Your name for the learning session"),
):
    """Start a new interactive learning session."""
    asyncio.run(cli.start_session(student_name))


@app.command() 
def ask(
    question: str = typer.Argument(..., help="Your educational question"),
    subject: Optional[str] = typer.Option(None, help="Subject area (math, science, music)"),
    topic: Optional[str] = typer.Option(None, help="Specific topic within the subject"),
    difficulty: Optional[str] = typer.Option(None, help="Difficulty level (elementary, middle, high, college)")
):
    """Ask a question to the educational agents."""
    
    # Convert string inputs to enums
    subject_enum = None
    if subject:
        try:
            subject_enum = SubjectType(subject.lower())
        except ValueError:
            console.print(f"❌ Invalid subject: {subject}. Use: math, science, or music", style="red")
            return
    
    difficulty_enum = None
    if difficulty:
        try:
            difficulty_enum = DifficultyLevel(difficulty.lower())
        except ValueError:
            console.print(f"❌ Invalid difficulty: {difficulty}. Use: elementary, middle, high, or college", style="red")
            return
    
    asyncio.run(cli.ask_question(question, subject_enum, topic, difficulty_enum))


@app.command()
def interactive():
    """Start an interactive chat session."""
    
    if not cli.current_session_id:
        student_name = Prompt.ask("👋 What's your name?")
        asyncio.run(cli.start_session(student_name))
    
    console.print("\n🎓 Interactive mode started! Type 'quit' to exit, 'help' for commands.\n")
    
    while True:
        try:
            user_input = Prompt.ask("🙋 Ask me anything").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("👋 Thanks for learning with us! Goodbye!", style="blue")
                break
            elif user_input.lower() == 'help':
                _show_help()
            elif user_input.lower() == 'agents':
                cli.show_agents()
            elif user_input.lower() == 'summary':
                cli.show_session_summary()
            elif user_input:
                asyncio.run(cli.ask_question(user_input))
            
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="blue")
            break
        except Exception as e:
            console.print(f"❌ Error: {str(e)}", style="red")


@app.command()
def agents():
    """Show information about available educational agents."""
    cli.show_agents()


@app.command()
def summary():
    """Show current session summary."""
    cli.show_session_summary()


def _show_help():
    """Show help information."""
    console.print(Panel(
        "🆘 **Available Commands:**\n\n"
        "• Just type your question naturally\n"
        "• `agents` - View available educational agents\n"
        "• `summary` - See your learning session summary\n"
        "• `help` - Show this help message\n"
        "• `quit` - Exit the interactive session\n\n"
        "**Example Questions:**\n"
        "• 'What is the quadratic formula?'\n"
        "• 'How do plants perform photosynthesis?'\n"
        "• 'Explain major and minor scales in music'\n"
        "• 'How does sound frequency relate to musical pitch?'",
        title="📚 Help",
        border_style="yellow"
    ))


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="blue")
    except Exception as e:
        console.print(f"❌ Unexpected error: {str(e)}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
