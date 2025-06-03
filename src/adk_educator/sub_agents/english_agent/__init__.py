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

"""English Agent: Specialized agent for English language, grammar, and writing education."""

import os
from pathlib import Path
from google.adk.agents import LlmAgent

MODEL = "gemini-2.5-flash-preview-05-20"

ENGLISH_AGENT_PROMPT = """
You are a specialized English Language Education Agent. Your role is to:

1. **Teach grammar rules** including syntax, punctuation, sentence structure, and parts of speech
2. **Analyze writing** for grammar errors, style improvements, and clarity
3. **Explain language concepts** such as tenses, voice, mood, and sentence types
4. **Provide writing guidance** for essays, creative writing, and academic papers
5. **Help with vocabulary** including word choice, synonyms, etymology, and usage
6. **Teach reading comprehension** strategies and literary analysis

Teaching Principles:
- Break down complex grammar rules into understandable parts
- Provide clear examples and counterexamples
- Explain the "why" behind grammar rules and conventions
- Use real-world writing contexts to demonstrate concepts
- Encourage proper usage while fostering creativity in expression
- Provide constructive feedback on writing samples

Specialties:
- Grammar correction and explanation
- Sentence structure and syntax analysis
- Punctuation rules and usage
- Writing style and clarity improvement
- Vocabulary building and word choice
- Reading comprehension strategies
- Essay and creative writing guidance

Always respond with clear explanations, helpful examples, and encouraging guidance that helps students improve their English language skills.
"""

def create_english_agent():
    """
    Create English agent: use local fine-tuned model if available, otherwise Gemini
    """
    # Check for local fine-tuned model
    finetuned_path = Path(__file__).parent.parent.parent.parent / "finetuning" / "english_agent_finetuned"
    
    if finetuned_path.exists():
        try:
            print("🎯 Found local fine-tuned English model, loading...")
            
            # Import here to avoid dependencies if not needed
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            class LocalEnglishAgent(LlmAgent):
                def __init__(self):
                    super().__init__(
                        name="english_agent_finetuned",
                        model="local-finetuned",
                        description="Fine-tuned English Language Education Agent (Local)",
                        instruction="Local fine-tuned English education agent",
                        output_key="english_response"
                    )
                    self._load_local_model(str(finetuned_path))
                
                def _load_local_model(self, model_path):
                    """Load the local fine-tuned model"""
                    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                    
                    # Load tokenizer and model
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    base_model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/DialoGPT-small",
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    self.model = PeftModel.from_pretrained(base_model, model_path)
                    self.model.eval()
                    print("✅ Local fine-tuned model loaded successfully")
                
                def __call__(self, prompt: str) -> str:
                    """Generate response using local model"""
                    try:
                        formatted_prompt = f"""<|im_start|>system
You are a specialized English Language Education Agent.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
                        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                inputs.input_ids,
                                max_new_tokens=200,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                        
                        response = self.tokenizer.decode(
                            outputs[0][len(inputs.input_ids[0]):], 
                            skip_special_tokens=True
                        ).strip()
                        
                        return response
                    except Exception as e:
                        print(f"❌ Local model error: {e}")
                        return f"Error with local model: {str(e)}"
            
            return LocalEnglishAgent()
            
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            print("🔄 Falling back to Gemini...")
    
    else:
        print("📝 Local fine-tuned model not found, using Gemini")
    
    # Fallback to Gemini
    return LlmAgent(
        name="english_agent",
        model=MODEL,
        description=(
            "Specialized agent for English language education covering grammar, "
            "writing, vocabulary, reading comprehension, and language instruction"
        ),
        instruction=ENGLISH_AGENT_PROMPT,
        output_key="english_response",
    )

english_agent = LlmAgent(
    name="english_agent",
    model=MODEL,
    description=(
        "Specialized agent for English language education covering grammar, "
        "writing, vocabulary, reading comprehension, and language instruction"
    ),
    instruction=ENGLISH_AGENT_PROMPT,
    output_key="english_response",
)
