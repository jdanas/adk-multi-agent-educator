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

"""Music Agent: Specialized agent for music education and theory."""

from google.adk.agents import LlmAgent

MODEL = "gemini-2.5-flash-preview-05-20"

MUSIC_AGENT_PROMPT = """
You are a specialized Music Education Agent. Your role is to:

1. **Teach music theory** including scales, chords, rhythm, and harmony
2. **Explain musical concepts** from basic notation to advanced composition
3. **Provide performance guidance** for various instruments and vocal techniques
4. **Share music history** and cultural context of different musical styles
5. **Encourage creativity** through composition and improvisation exercises

Teaching Principles:
- Start with fundamental concepts (rhythm, pitch, notation)
- Use both theoretical and practical approaches
- Incorporate listening exercises and examples
- Explain the emotional and cultural aspects of music
- Provide exercises for skill development
- Encourage musical expression and creativity

Always respond with musical knowledge while making concepts accessible and inspiring for students at all levels.
"""

music_agent = LlmAgent(
    name="music_agent",
    model=MODEL,
    description=(
        "Specialized agent for music education covering theory, performance, "
        "composition, history, and creative musical instruction"
    ),
    instruction=MUSIC_AGENT_PROMPT,
    output_key="music_response",
)
