"""
Evaluation script for fine-tuned English Agent on Mac M4
Tests grammar correction, vocabulary, and writing assistance capabilities
"""

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict

class EnglishAgentEvaluator:
    """Evaluator for fine-tuned English education agent"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = self._setup_device()
        self.model, self.tokenizer = self._load_model()
        
    def _setup_device(self):
        """Setup device for evaluation"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _load_model(self):
        """Load the fine-tuned model"""
        print(f"🔄 Loading model from {self.model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, self.model_path)
        model.eval()
        
        return model, tokenizer
    
    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate response from the fine-tuned model"""
        
        formatted_prompt = f"""<|im_start|>system
You are a specialized English Language Education Agent.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=400
        )
        
        if self.device.type == "mps":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
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
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        return response.strip()

def create_evaluation_dataset():
    """Create comprehensive test cases for evaluation"""
    
    test_cases = [
        # Grammar Correction Tests
        {
            "category": "Grammar Correction",
            "input": "Fix this grammar: 'The group of students are studying hard.'",
            "expected_keywords": ["is studying", "collective noun", "singular verb"],
            "difficulty": "intermediate"
        },
        {
            "category": "Grammar Correction", 
            "input": "What's wrong with: 'Between you and I, this is confidential.'",
            "expected_keywords": ["between you and me", "object pronoun", "preposition"],
            "difficulty": "advanced"
        },
        {
            "category": "Grammar Correction",
            "input": "Fix: 'Me and Sarah went to the movies yesterday.'",
            "expected_keywords": ["Sarah and I", "subject pronoun", "order"],
            "difficulty": "basic"
        },
        
        # Vocabulary Tests
        {
            "category": "Vocabulary",
            "input": "Explain the difference between affect and effect.",
            "expected_keywords": ["affect", "verb", "effect", "noun", "influence", "result"],
            "difficulty": "intermediate"
        },
        {
            "category": "Vocabulary",
            "input": "When should I use 'who' vs 'whom'?",
            "expected_keywords": ["who", "subject", "whom", "object", "he/him test"],
            "difficulty": "advanced"
        },
        {
            "category": "Vocabulary",
            "input": "What's the difference between 'less' and 'fewer'?",
            "expected_keywords": ["fewer", "countable", "less", "uncountable"],
            "difficulty": "intermediate"
        },
        
        # Writing Improvement Tests
        {
            "category": "Writing Improvement",
            "input": "Help me improve: 'The thing was really good and stuff.'",
            "expected_keywords": ["specific", "vague", "descriptive", "concrete"],
            "difficulty": "basic"
        },
        {
            "category": "Writing Improvement",
            "input": "How do I write a strong thesis statement for an essay about climate change?",
            "expected_keywords": ["thesis", "specific", "arguable", "preview", "position"],
            "difficulty": "advanced"
        },
        
        # Reading Comprehension Tests
        {
            "category": "Reading Comprehension",
            "input": "How do I identify the main idea in a paragraph?",
            "expected_keywords": ["topic sentence", "supporting details", "main point"],
            "difficulty": "intermediate"
        }
    ]
    
    return test_cases

def evaluate_response_quality(response: str, expected_keywords: List[str]) -> Dict:
    """Evaluate the quality of a response"""
    
    response_lower = response.lower()
    
    # Check for expected keywords
    keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
    keyword_score = keyword_matches / len(expected_keywords)
    
    # Check response structure (educational qualities)
    structure_score = 0
    structure_checks = [
        ("examples" in response_lower, 0.2),  # Provides examples
        ("rule" in response_lower or "explanation" in response_lower, 0.3),  # Explains rules
        ("trick" in response_lower or "tip" in response_lower, 0.2),  # Memory aids
        ("practice" in response_lower or "try" in response_lower, 0.1),  # Encourages practice
        (len(response) > 100, 0.2)  # Adequate length
    ]
    
    for check, weight in structure_checks:
        if check:
            structure_score += weight
    
    # Overall quality score
    overall_score = (keyword_score * 0.6) + (structure_score * 0.4)
    
    return {
        "keyword_score": keyword_score,
        "structure_score": structure_score,
        "overall_score": overall_score,
        "keyword_matches": keyword_matches,
        "total_keywords": len(expected_keywords),
        "response_length": len(response)
    }

def run_comprehensive_evaluation(model_path: str):
    """Run comprehensive evaluation of the fine-tuned model"""
    
    print("🧪 Starting Comprehensive English Agent Evaluation")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("💡 Run finetune_english_agent_mac.py first")
        return
    
    # Initialize evaluator
    evaluator = EnglishAgentEvaluator(model_path)
    
    # Get test cases
    test_cases = create_evaluation_dataset()
    
    # Track results
    results = []
    category_scores = {}
    
    print(f"📊 Running {len(test_cases)} test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔍 Test {i}/{len(test_cases)}: {test_case['category']}")
        print(f"❓ Question: {test_case['input']}")
        
        # Generate response
        try:
            response = evaluator.generate_response(test_case['input'])
            
            # Evaluate response
            evaluation = evaluate_response_quality(response, test_case['expected_keywords'])
            
            # Store results
            result = {
                **test_case,
                'response': response,
                'evaluation': evaluation
            }
            results.append(result)
            
            # Track category performance
            category = test_case['category']
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(evaluation['overall_score'])
            
            # Display results
            print(f"📈 Overall Score: {evaluation['overall_score']:.2f}")
            print(f"🎯 Keywords Found: {evaluation['keyword_matches']}/{evaluation['total_keywords']}")
            print(f"📝 Response Length: {evaluation['response_length']} chars")
            
            # Show response preview
            response_preview = response[:200] + "..." if len(response) > 200 else response
            print(f"🤖 Response: {response_preview}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            print("-" * 50)
    
    # Calculate overall statistics
    print("\n📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    if results:
        overall_scores = [r['evaluation']['overall_score'] for r in results]
        avg_score = sum(overall_scores) / len(overall_scores)
        
        print(f"🎯 Overall Average Score: {avg_score:.3f}")
        print(f"📈 Best Score: {max(overall_scores):.3f}")
        print(f"📉 Lowest Score: {min(overall_scores):.3f}")
        
        # Category breakdown
        print(f"\n📋 Category Performance:")
        for category, scores in category_scores.items():
            avg_category_score = sum(scores) / len(scores)
            print(f"   {category}: {avg_category_score:.3f} ({len(scores)} tests)")
        
        # Difficulty analysis
        difficulty_scores = {}
        for result in results:
            diff = result['difficulty']
            if diff not in difficulty_scores:
                difficulty_scores[diff] = []
            difficulty_scores[diff].append(result['evaluation']['overall_score'])
        
        print(f"\n🎚️ Difficulty Performance:")
        for difficulty, scores in difficulty_scores.items():
            avg_diff_score = sum(scores) / len(scores)
            print(f"   {difficulty.title()}: {avg_diff_score:.3f} ({len(scores)} tests)")
    
    # Save detailed results
    results_file = "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if avg_score >= 0.8:
        print("   ✅ Model performs excellently! Ready for production.")
    elif avg_score >= 0.6:
        print("   ⚠️ Model performs well but could use more training data.")
    else:
        print("   ❌ Model needs more training. Consider:")
        print("      • More diverse training examples")
        print("      • Longer training epochs")
        print("      • Better prompt formatting")

def compare_with_baseline():
    """Compare fine-tuned model with baseline (optional)"""
    print("\n🔄 Baseline Comparison")
    print("=" * 30)
    print("💡 To compare with baseline:")
    print("1. Save responses from original model")
    print("2. Run same evaluation")
    print("3. Compare scores")

if __name__ == "__main__":
    model_path = "./english_agent_finetuned"
    
    print("🎓 English Agent Evaluation Tool")
    print("=" * 40)
    
    # Run evaluation
    run_comprehensive_evaluation(model_path)
    
    # Optional baseline comparison
    compare_with_baseline()
    
    print("\n🚀 Evaluation complete!")
    print("📈 Use results to improve your fine-tuning process.")
