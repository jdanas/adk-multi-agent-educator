"""
Integration Guide: Fine-tuned English Agent with ADK (Mac M4 Compatible)

This script shows how to integrate your Mac-trained fine-tuned English agent
back into the ADK Multi-Agent Educational system.
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from google.adk.agents import LlmAgent
from typing import Optional

class MacFineTunedEnglishAgent:
    """
    Mac-compatible wrapper class to integrate fine-tuned model with ADK
    """
    
    def __init__(self, model_path: str):
        """Initialize the fine-tuned English agent for Mac"""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device()
        self._load_model()
    
    def _setup_device(self):
        """Setup device for Mac M4"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"✅ Using Apple Silicon MPS: {device}")
        else:
            device = torch.device("cpu")
            print(f"⚠️ Using CPU: {device}")
        return device
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            print(f"🔄 Loading fine-tuned model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-small",  # Same base model used in training
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            print("✅ Fine-tuned model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response using the fine-tuned model"""
        
        # Format the prompt for the fine-tuned model
        formatted_prompt = f"""<|im_start|>system
You are a specialized English Language Education Agent. Your role is to help students with grammar, writing, vocabulary, and language concepts. Always provide clear explanations with examples.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to device if using MPS
        if self.device.type == "mps":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"

def create_adk_compatible_english_agent(model_path: Optional[str] = None):
    """
    Create an ADK-compatible English agent using either:
    1. Fine-tuned local model (if model_path provided and exists)
    2. Original Gemini model (fallback)
    """
    
    if model_path and os.path.exists(model_path):
        # Use fine-tuned model
        print("🎯 Using fine-tuned English agent (Mac M4 compatible)")
        
        class CustomEnglishAgent(LlmAgent):
            def __init__(self):
                # Initialize the fine-tuned model wrapper
                try:
                    self.fine_tuned_agent = MacFineTunedEnglishAgent(model_path)
                    
                    # Initialize parent with basic config
                    super().__init__(
                        name="english_agent_finetuned_mac",
                        model="custom-mac-finetuned",
                        description="Fine-tuned English Language Education Agent (Mac M4) specializing in grammar, writing, vocabulary, and language instruction",
                        instruction="Mac-compatible fine-tuned agent for English education",
                        output_key="english_response"
                    )
                    print("✅ Custom fine-tuned agent initialized")
                    
                except Exception as e:
                    print(f"❌ Failed to initialize fine-tuned agent: {e}")
                    raise
            
            def __call__(self, prompt: str) -> str:
                """Override the call method to use our fine-tuned model"""
                try:
                    return self.fine_tuned_agent.generate_response(prompt)
                except Exception as e:
                    return f"I apologize, but I encountered an error: {str(e)}"
        
        try:
            return CustomEnglishAgent()
        except Exception as e:
            print(f"❌ Failed to create custom agent, falling back to Gemini: {e}")
            # Fall through to fallback
    
    else:
        # Fallback to original Gemini-based agent
        print("🔄 Using original Gemini-based English agent (fallback)")
    
    # Fallback Gemini agent
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

def test_integration():
    """Test the integration with sample questions"""
    
    # Test questions for the English agent
    test_questions = [
        "Fix this grammar: 'Me and my friend went to store.'",
        "Explain the difference between affect and effect",
        "Help me write a thesis statement about climate change",
        "What's wrong with: 'Between you and I, this is difficult'?"
    ]
    
    print("🧪 Testing Fine-tuned English Agent Integration (Mac M4)")
    print("=" * 60)
    
    # Try to create the fine-tuned agent
    try:
        finetuned_path = "./english_agent_finetuned"
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
                    response_preview = response[:150] + "..." if len(response) > 150 else response
                    print(f"✅ Fine-tuned response: {response_preview}")
                else:
                    print("📝 Using standard ADK agent response")
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

def update_root_agent_example():
    """Show how to update root_agent.py for Mac fine-tuned model"""
    
    integration_code = '''
# Updated root_agent.py with Mac fine-tuned English agent

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from .adk_educator import prompt
from .adk_educator.sub_agents.math_agent import math_agent
from .adk_educator.sub_agents.science_agent import science_agent  
from .adk_educator.sub_agents.music_agent import music_agent

# Import the Mac-compatible fine-tuned English agent
from .finetuning.integrate_finetuned_mac import create_adk_compatible_english_agent

MODEL = "gemini-2.5-flash-preview-05-20"

# Create English agent (Mac fine-tuned if available)
english_agent = create_adk_compatible_english_agent(
    model_path="./finetuning/english_agent_finetuned"
)

educational_coordinator = LlmAgent(
    name="educational_coordinator",
    model=MODEL,
    description=(
        "Coordinating educational support across mathematics, science, music, and English, "
        "routing student questions to specialized agents (including Mac fine-tuned English agent), "
        "and providing comprehensive learning assistance"
    ),
    instruction=prompt.EDUCATIONAL_COORDINATOR_PROMPT,
    output_key="educational_response",
    tools=[
        AgentTool(agent=math_agent),
        AgentTool(agent=science_agent),
        AgentTool(agent=music_agent),
        AgentTool(agent=english_agent),  # Now using Mac fine-tuned model!
    ],
)

root_agent = educational_coordinator
'''
    
    print("📝 Mac Integration Code Example:")
    print("=" * 50)
    print(integration_code)
    print("=" * 50)
    
    return integration_code

if __name__ == "__main__":
    print("🔗 Mac Fine-tuned English Agent Integration Guide")
    print("=" * 60)
    
    # Show integration code
    update_root_agent_example()
    
    # Test the integration
    test_integration()
    
    print("\n📋 Mac Integration Steps:")
    print("1. ✅ Fine-tune your model using finetune_english_agent_mac.py")
    print("2. ✅ Copy this integration code to your project") 
    print("3. ✅ Update your root_agent.py imports")
    print("4. ✅ Test with 'adk web' command")
    print("5. ✅ Monitor performance and iterate")
    
    print("\n🎯 Benefits of Mac Fine-tuning:")
    print("• More consistent educational responses")
    print("• Better grammar explanation structure")
    print("• Specialized vocabulary for teaching")
    print("• Reduced hallucinations in grammar rules")
    print("• Efficient inference on Apple Silicon")
    print("• No dependency on external GPUs or cloud services")
    
    print("\n🚀 Ready to deploy your Mac fine-tuned English agent!")
