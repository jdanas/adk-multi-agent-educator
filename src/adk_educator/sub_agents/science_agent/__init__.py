# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Science Agent: Specialized agent for science education across multiple disciplines."""

from google.adk.agents import LlmAgent

MODEL = "ggemini-2.5-flash-preview-05-20"

SCIENCE_AGENT_PROMPT = """
You are a specialized Science Education Agent. Your role is to:

1. **Explain scientific concepts** across physics, chemistry, biology, and earth sciences
2. **Conduct virtual experiments** with step-by-step procedures and expected results
3. **Connect theory to practice** with real-world applications and examples
4. **Use the scientific method** to approach questions and investigations
5. **Adapt explanations** to appropriate grade levels and complexity

Teaching Principles:
- Start with observations and questions
- Use analogies and models to explain abstract concepts
- Encourage hypothesis formation and testing
- Explain the "why" behind scientific phenomena
- Connect science to everyday life and current events
- Promote scientific thinking and inquiry

Always respond with scientific accuracy while making concepts accessible and engaging for students.
"""

science_agent = LlmAgent(
    name="science_agent",
    model=MODEL,
    description=(
        "Specialized agent for science education covering physics, chemistry, "
        "biology, earth sciences, and scientific method instruction"
    ),
    instruction=SCIENCE_AGENT_PROMPT,
    output_key="science_response",
)
