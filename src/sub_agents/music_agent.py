# Copyright 2025 ADK Multi-Agent Educator
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

"""Music agent for the ADK Multi-Agent Educator."""

try:
    from google.adk import Agent
    from . import prompt

    MODEL = "gemini-2.5-flash-preview-05-20"

    music_agent = Agent(
        model=MODEL,
        name="music_agent",
        instruction=prompt.MUSIC_AGENT_PROMPT,
        output_key="music_guidance",
    )
except ImportError:
    # Graceful fallback if ADK is not available
    music_agent = None
