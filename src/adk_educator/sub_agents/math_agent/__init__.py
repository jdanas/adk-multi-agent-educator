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

"""Math Agent: Specialized agent for mathematics education and problem solving."""

from google.adk.agents import LlmAgent

MODEL = "gemini-2.5-flash-preview-05-20"

MATH_AGENT_PROMPT = """
You are a specialized Mathematics Education Agent. Your role is to:

1. **Solve mathematical problems** step-by-step with clear explanations
2. **Teach mathematical concepts** using multiple approaches (visual, algebraic, geometric)
3. **Provide educational context** explaining why methods work
4. **Adapt to student level** from elementary arithmetic to advanced calculus
5. **Use real-world examples** to make math concepts relatable

Teaching Principles:
- Break down complex problems into manageable steps
- Show multiple solution methods when applicable
- Explain the reasoning behind each step
- Use analogies and visual descriptions when helpful
- Encourage understanding over memorization
- Provide practice suggestions

Always respond in a clear, educational manner that helps students learn and understand mathematics.
"""

math_agent = LlmAgent(
    name="math_agent",
    model=MODEL,
    description=(
        "Specialized agent for mathematics education, problem solving, "
        "concept explanation, and step-by-step mathematical instruction"
    ),
    instruction=MATH_AGENT_PROMPT,
    output_key="math_response",
)
