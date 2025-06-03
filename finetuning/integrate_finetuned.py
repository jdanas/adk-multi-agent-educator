"""
Integration Guide: Fine-tuned English Agent with ADK

This script shows how to integrate your fine-tuned English agent
back into the ADK Multi-Agent Educational system.
"""

from google.adk.agents import LlmAgent
import torch
import os
from typing import Optional

class FineTunedEnglishAgent:
    """
    Wrapper class to integrate fine-tuned model with ADK
    """
    
    def __init__(self, model_path: str):
        """Initialize the fine-tuned English agent"""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            from unsloth import FastLanguageModel
            
            print(f"🔄 Loading fine-tuned model from {self.model_path}")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=None,
            )
            
            # Enable inference mode for faster generation
            FastLanguageModel.for_inference(self.model)
            
            print("✅ Fine-tuned model loaded successfully")
            
        except ImportError:
            print("⚠️ Unsloth not available. Install with: pip install unsloth")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using the fine-tuned model"""
        
        # Format the prompt for the fine-tuned model
        messages = [
            {
                "role": "system", 
                "content": "You are a specialized English Language Education Agent. Your role is to help students with grammar, writing, vocabulary, and language concepts. Always provide clear explanations with examples."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        )
        
        return response.strip()

def create_adk_compatible_english_agent(model_path: Optional[str] = None):
    """
    Create an ADK-compatible English agent using either:
    1. Fine-tuned local model (if model_path provided)
    2. Original Gemini model (fallback)
    """
    
    if model_path and os.path.exists(model_path):
        # Use fine-tuned model
        print("🎯 Using fine-tuned English agent")
        
        class CustomEnglishAgent(LlmAgent):
            def __init__(self):
                # Initialize the fine-tuned model wrapper
                self.fine_tuned_agent = FineTunedEnglishAgent(model_path)
                
                # Initialize parent with basic config
                super().__init__(
                    name="english_agent_finetuned",
                    model="custom",  # This won't be used since we override
                    description="Fine-tuned English Language Education Agent specializing in grammar, writing, vocabulary, and language instruction",
                    instruction="Custom fine-tuned agent for English education",
                    output_key="english_response"
                )
            
            def __call__(self, prompt: str) -> str:
                """Override the call method to use our fine-tuned model"""
                return self.fine_tuned_agent.generate_response(prompt)
        
        return CustomEnglishAgent()
    
    else:
        # Fallback to original Gemini-based agent
        print("🔄 Using original Gemini-based English agent")
        
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

Always respond with clear explanations, helpful examples, and encouraging guidance.
"""

        return LlmAgent(
            name="english_agent",
            model="gemini-2.5-pro-preview-05-06",
            description="English Language Education Agent covering grammar, writing, vocabulary, reading comprehension, and language instruction",
            instruction=ENGLISH_AGENT_PROMPT,
            output_key="english_response",
        )

def update_root_agent_with_finetuned():
    """
    Example of how to update your root_agent.py to use the fine-tuned model
    """
    
    integration_code = '''
# Updated root_agent.py with fine-tuned English agent

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from .adk_educator import prompt
from .adk_educator.sub_agents.math_agent import math_agent
from .adk_educator.sub_agents.science_agent import science_agent  
from .adk_educator.sub_agents.music_agent import music_agent

# Import the fine-tuned English agent
from .finetuning.integrate_finetuned import create_adk_compatible_english_agent

MODEL = "gemini-2.5-flash-preview-05-20"

# Create English agent (fine-tuned if available)
english_agent = create_adk_compatible_english_agent(
    model_path="./finetuning/english_agent_finetuned/final_model"
)

educational_coordinator = LlmAgent(
    name="educational_coordinator",
    model=MODEL,
    description=(
        "Coordinating educational support across mathematics, science, music, and English, "
        "routing student questions to specialized agents (including fine-tuned English agent), "
        "and providing comprehensive learning assistance"
    ),
    instruction=prompt.EDUCATIONAL_COORDINATOR_PROMPT,
    output_key="educational_response",
    tools=[
        AgentTool(agent=math_agent),
        AgentTool(agent=science_agent),
        AgentTool(agent=music_agent),
        AgentTool(agent=english_agent),  # Now using fine-tuned model!
    ],
)

root_agent = educational_coordinator
'''
    
    print("📝 Integration Code Example:")
    print("=" * 50)
    print(integration_code)
    print("=" * 50)
    
    return integration_code

def test_integration():
    """Test the integration with sample questions"""
    
    # Test questions for the English agent
    test_questions = [
        "Fix this grammar: 'Me and my friend went to store.'",
        "Explain the difference between affect and effect",
        "Help me write a thesis statement about climate change",
        "What's wrong with: 'Between you and I, this is difficult'?"
    ]
    
    print("🧪 Testing Fine-tuned English Agent Integration")
    print("=" * 50)
    
    # Try to create the fine-tuned agent
    try:
        finetuned_path = "./english_agent_finetuned/final_model"
        if os.path.exists(finetuned_path):
            agent = create_adk_compatible_english_agent(finetuned_path)
            print("✅ Fine-tuned agent loaded successfully")
        else:
            agent = create_adk_compatible_english_agent()
            print("⚠️ Fine-tuned model not found, using fallback")
        
        print(f"🤖 Agent type: {type(agent)}")
        print(f"📝 Agent name: {agent.name}")
        
        # Test with sample questions
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 Test {i}: {question}")
            try:
                if hasattr(agent, 'fine_tuned_agent'):
                    response = agent.fine_tuned_agent.generate_response(question)
                    print(f"✅ Fine-tuned response generated (length: {len(response)} chars)")
                else:
                    print("📝 Using standard ADK agent response")
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

if __name__ == "__main__":
    print("🔗 Fine-tuned English Agent Integration Guide")
    print("=" * 60)
    
    # Show integration code
    update_root_agent_with_finetuned()
    
    # Test the integration
    test_integration()
    
    print("\n📋 Integration Steps:")
    print("1. ✅ Fine-tune your model using finetune_english_agent.py")
    print("2. ✅ Copy this integration code to your project") 
    print("3. ✅ Update your root_agent.py imports")
    print("4. ✅ Test with 'adk web' command")
    print("5. ✅ Monitor performance and iterate")
    
    print("\n🎯 Benefits of Fine-tuning:")
    print("• More consistent educational responses")
    print("• Better grammar explanation structure")
    print("• Specialized vocabulary for teaching")
    print("• Reduced hallucinations in grammar rules")
    print("• 2x faster inference with Unsloth optimizations")
    
    print("\n🚀 Ready to deploy your fine-tuned English agent!")
