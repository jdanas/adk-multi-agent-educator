# ADK Multi-Agent Educator

A sophisticated multi-agent educational system built with Google's ADK framework, featuring specialized AI tutors for **Math**, **Science**, **Music**, and **English**. Each agent is expertly trained in their subject domain and works together through an intelligent coordinator to provide comprehensive, personalized educational support.

## ✨ Features

- **🎯 Four Specialized Agents**: Expert tutors for Math, Science, Music, and English
- **🤝 Intelligent Coordination**: Smart routing to the most appropriate agent
- **✍️ Grammar & Writing Support**: Comprehensive English language assistance
- **📊 ADK Integration**: Built on Google's Agent Development Kit
- **🚀 Minimal Dependencies**: Only requires `google-adk`
- **🌐 Web Interface**: Easy access through `adk web` command

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd adk-multi-agent-educator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Launch ADK Web Interface:**
```bash
adk web
```

4. **Start learning!** Ask questions like:
   - "What is 2+2?" (Math Agent)
   - "How does photosynthesis work?" (Science Agent)
   - "Explain major scales" (Music Agent)
   - "Fix this sentence grammar" (English Agent)

## 🎓 Agent Capabilities

### 📐 **Math Agent**
- **Core Topics**: Algebra, Geometry, Calculus, Statistics, Arithmetic
- **Teaching Style**: Step-by-step problem solving with clear explanations
- **Special Features**: Multiple solution methods, real-world applications
- **Example**: "How do I solve x² + 5x + 6 = 0?"

### 🔬 **Science Agent**
- **Core Topics**: Physics, Chemistry, Biology, Earth Sciences
- **Teaching Style**: Scientific method approach with experiments
- **Special Features**: Virtual experiments, theory-to-practice connections
- **Example**: "Why do objects fall at the same rate in a vacuum?"

### 🎵 **Music Agent**
- **Core Topics**: Music theory, composition, performance, history
- **Teaching Style**: Both theoretical and practical approaches
- **Special Features**: Cultural context, creative exercises
- **Example**: "What's the difference between major and minor chords?"

### ✍️ **English Agent** *New!*
- **Core Topics**: Grammar, writing, vocabulary, reading comprehension
- **Teaching Style**: Clear explanations with examples and practice
- **Special Features**: Writing feedback, style improvement, language rules
- **Example**: "What's the difference between 'affect' and 'effect'?"

## 🏗️ Architecture

```
src/
├── root_agent.py                    # Main ADK coordinator
└── adk_educator/
    ├── prompt.py                    # Coordination prompts
    └── sub_agents/                  # Specialized agents
        ├── math_agent/
        ├── science_agent/
        ├── music_agent/
        └── english_agent/
```

The system uses Google's ADK `LlmAgent` and `AgentTool` to create a coordinator that intelligently routes student questions to the most appropriate specialist agent.

## 💡 How It Works

1. **Student asks a question** through the ADK web interface
2. **Educational Coordinator** analyzes the question and determines the best agent
3. **Specialist Agent** provides expert assistance in their domain
4. **Response** is delivered with educational context and follow-up suggestions

## 🔧 Technical Details

### Requirements
- **Google ADK**: `google-adk>=0.1.0`
- **Python**: 3.8+

### Agent Structure
Each specialist agent is built using:
```python
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="agent_name",
    model="gemini-2.5-pro-preview-05-06",
    description="Agent specialization",
    instruction="Detailed teaching prompt",
    output_key="response_key"
)
```

### Coordinator Pattern
The main coordinator uses `AgentTool` to access specialists:
```python
educational_coordinator = LlmAgent(
    name="educational_coordinator",
    tools=[
        AgentTool(agent=math_agent),
        AgentTool(agent=science_agent),
        AgentTool(agent=music_agent),
        AgentTool(agent=english_agent),
    ]
)
```

## 🎯 Example Interactions

### Math Problem Solving
**Student**: "I need help with calculus derivatives"  
**Coordinator**: Routes to Math Agent  
**Math Agent**: Provides step-by-step derivative rules with examples

### Grammar Correction
**Student**: "Can you fix this sentence: 'Me and my friend went to store'"  
**Coordinator**: Routes to English Agent  
**English Agent**: Corrects to "My friend and I went to the store" with explanation

### Science Concepts
**Student**: "How do solar panels work?"  
**Coordinator**: Routes to Science Agent  
**Science Agent**: Explains photovoltaic effect with practical applications

### Music Theory
**Student**: "What makes a chord progression sound good?"  
**Coordinator**: Routes to Music Agent  
**Music Agent**: Explains harmony, tension, and resolution with examples

## 🔮 Future Enhancements

- **Additional Subjects**: History, Art, Computer Science agents
- **Multimodal Support**: Image analysis, audio processing
- **Learning Analytics**: Progress tracking and personalized recommendations
- **Collaborative Features**: Multi-student sessions and peer learning
- **Assessment Tools**: Quiz generation and automated grading

## 📄 License

Copyright 2025 Google LLC - Licensed under the Apache License, Version 2.0

## 🤝 Contributing

This project follows Google's ADK best practices and the academic research patterns for multi-agent educational systems.

---

**Ready to start learning?** Just run `adk web` and ask any educational question! 🚀
