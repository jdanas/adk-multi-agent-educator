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

"""Multi-Agent Educational System: Mathematics, Science, Music, and English education with specialized routing."""

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from .adk_educator import prompt
from .adk_educator.sub_agents.math_agent import math_agent
from .adk_educator.sub_agents.science_agent import science_agent
from .adk_educator.sub_agents.music_agent import music_agent
from .adk_educator.sub_agents.english_agent import english_agent

MODEL = "gemini-2.5-flash-preview-05-20"


educational_coordinator = LlmAgent(
    name="educational_coordinator",
    model=MODEL,
    description=(
        "Coordinating educational support across mathematics, science, music, and English, "
        "routing student questions to specialized agents, and providing "
        "comprehensive learning assistance"
    ),
    instruction=prompt.EDUCATIONAL_COORDINATOR_PROMPT,
    output_key="educational_response",
    tools=[
        AgentTool(agent=math_agent),
        AgentTool(agent=science_agent),
        AgentTool(agent=music_agent),
        AgentTool(agent=english_agent),
    ],
)

root_agent = educational_coordinator
