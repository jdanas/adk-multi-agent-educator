# ADK Multi-Agent Educator

A multi-agent educational system using Google's ADK that provides specialized tutoring in Math, Science, and Music. Each agent is expertly trained in their subject and can collaborate with other agents to provide comprehensive educational support.

## Features

- **Specialized Agents**: Three expert tutors for Math, Science, and Music
- **Collaborative Learning**: Agents work together on interdisciplinary questions
- **Adaptive Difficulty**: Adjusts to student's learning level
- **Session Management**: Tracks learning progress and preferences
- **Multiple Interfaces**: CLI, Web API, and programmatic access

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd adk-multi-agent-educator
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Google API key
```

4. Get your Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Usage

#### Command Line Interface
```bash
python -m src.adk_educator.cli
```

#### Programmatic Usage
```python
from src.adk_educator.agents.math_agent import MathAgent
from src.adk_educator.config import StudentRequest, SubjectType, DifficultyLevel

# Create a math agent
math_agent = MathAgent()

# Ask a question
request = StudentRequest(
    session_id="demo",
    student_id="Alice",
    subject=SubjectType.MATH,
    topic="algebra",
    difficulty=DifficultyLevel.HIGH,
    specific_question="How do I solve quadratic equations?"
)

response = await math_agent.process_request(request)
print(response.response_text)
```

## Agent Capabilities

### 📐 Math Agent
- Algebra, Geometry, Calculus, Statistics
- Step-by-step problem solving
- Visual explanations and diagrams
- Real-world applications

### 🔬 Science Agent
- Physics, Chemistry, Biology, Earth Science
- Experiment suggestions
- Scientific method guidance
- Current research connections

### 🎵 Music Agent
- Music theory, composition, history
- Cultural context and diversity
- Practice exercises
- Performance techniques
