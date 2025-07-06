# KyleBot: A GPT-2 Chatbot for Learning
# Fixed version that resolves parameter passing issues

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import re

# Load the pre-trained GPT-2 model and tokenizer
print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the model to evaluation mode (no training)
model.eval()

# Add a special token for the end of text
tokenizer.pad_token = tokenizer.eos_token

print("✅ GPT-2 loaded successfully!")
print(f"Model parameters: {model.num_parameters():,}")

def generate_response_greedy(prompt, max_new_tokens=50, **kwargs):
    """Greedy decoding: Always picks the most likely next word"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

def generate_response_sampling(prompt, max_new_tokens=50, temperature=0.8, top_k=50, **kwargs):
    """
    Sampling with temperature and top-k: More creative and diverse responses
    - temperature: Controls randomness (higher = more random)
    - top_k: Only considers the top k most likely words
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

def generate_response_beam_search(prompt, max_new_tokens=50, num_beams=5, **kwargs):
    """Beam search: Explores multiple possible sequences"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Set default values for beam search
    beam_kwargs = {
        'no_repeat_ngram_size': 2,
        'repetition_penalty': 1.2
    }
    # Update with any provided kwargs
    beam_kwargs.update(kwargs)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            pad_token_id=tokenizer.eos_token_id,
            **beam_kwargs
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

class KyleBot:
    def __init__(self, name="KyleBot"):
        self.name = name
        self.conversation_history = []
        self.generation_method = "sampling"  # Default method
        
    def add_to_history(self, user_input, bot_response):
        """Keep track of conversation for context"""
        self.conversation_history.append({
            "user": user_input,
            "bot": bot_response
        })
        
    def create_context_prompt(self, user_input):
        """Create a prompt with conversation history for context"""
        prompt = f"You are {self.name}, a knowledgeable AI. Provide a clear and concise definition of the user's topic.\n"
        if len(self.conversation_history) > 0:
            recent_history = self.conversation_history[-2:]
            context = "\n".join([
                f"User: {exchange['user']}\n{self.name}: {exchange['bot']}"
                for exchange in recent_history
            ])
            prompt += context + "\n"
        prompt += f"User: {user_input}\n{self.name}: "
        return prompt
    
    def generate_response(self, user_input, method=None, **kwargs):
        """Generate a response using the specified method"""
        method = method or self.generation_method
        prompt = self.create_context_prompt(user_input)
        
        if method == "greedy":
            response = generate_response_greedy(prompt, **kwargs)
        elif method == "sampling":
            response = generate_response_sampling(prompt, **kwargs)
        elif method == "beam":
            response = generate_response_beam_search(prompt, **kwargs)
        else:
            response = generate_response_sampling(prompt, **kwargs)
    
        response = self.clean_response(response)
        self.add_to_history(user_input, response)
        return response
    
    def clean_response(self, response, max_chars=500):
        """Clean up the generated response"""
        response = response.strip()
        response = re.sub(rf"^{self.name}:\s*|^User:\s*|\n{self.name}:.*", "", response)
        if len(response) > max_chars:
            for char in ['.', '!', '?']:
                if char in response[:max_chars]:
                    response = response[:response.index(char) + 1]
                    break
        return response
    
    def set_generation_method(self, method):
        """Change the generation method"""
        valid_methods = ["greedy", "sampling", "beam"]
        if method in valid_methods:
            self.generation_method = method
            print(f"✅ Generation method set to: {method}")
        else:
            print(f"❌ Invalid method. Choose from: {valid_methods}")

# Create our chatbot instance
kylebot = KyleBot()
print("✅ KyleBot created and ready to chat!")

def test_generation_methods():
    """Test all three generation methods"""
    test_prompt = "What is artificial intelligence?"
    
    print("🤖 Testing different generation methods:")
    print("=" * 50)
    
    # Test greedy decoding
    print("\n1. GREEDY DECODING (always picks most likely word):")
    response = kylebot.generate_response(test_prompt, method="greedy", max_new_tokens=100, no_repeat_ngram_size=2, repetition_penalty=1.2)
    print(f"Response: {response}")
    
    # Test sampling
    print("\n2. SAMPLING (more creative, uses temperature and top-k):")
    response = kylebot.generate_response(test_prompt, method="sampling", max_new_tokens=100, temperature=0.7, top_k=30)
    print(f"Response: {response}")
    
    # Test beam search
    print("\n3. BEAM SEARCH (explores multiple possibilities):")
    response = kylebot.generate_response(test_prompt, method="beam", max_new_tokens=100, num_beams=5, no_repeat_ngram_size=2, repetition_penalty=1.2)
    print(f"Response: {response}")
    
    print("\n" + "=" * 50)
    print("💡 Notice how each method produces different styles of responses!")

def chat_with_kylebot():
    """Interactive chat interface"""
    print("🤖 Welcome to KyleBot! Let's chat!")
    print("💡 Commands:")
    print("   - Type 'quit' to exit")
    print("   - Type 'method: [greedy/sampling/beam]' to change generation method")
    print("   - Type 'history' to see conversation history")
    print("   - Type 'help' for this message")
    print("   - Type 'test' to run generation method tests")
    print("=" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("👋 Goodbye! Thanks for chatting with KyleBot!")
                break
            elif user_input.lower() == 'help':
                print("💡 Commands:")
                print("   - Type 'quit' to exit")
                print("   - Type 'method: [greedy/sampling/beam]' to change generation method")
                print("   - Type 'history' to see conversation history")
                print("   - Type 'help' for this message")
                print("   - Type 'test' to run generation method tests")
                continue
            elif user_input.lower() == 'test':
                test_generation_methods()
                continue
            elif user_input.lower() == 'history':
                if kylebot.conversation_history:
                    print("\n📜 Conversation History:")
                    for i, exchange in enumerate(kylebot.conversation_history, 1):
                        print(f"{i}. You: {exchange['user']}")
                        print(f"   KyleBot: {exchange['bot']}")
                else:
                    print("📜 No conversation history yet.")
                continue
            elif user_input.lower().startswith('method:'):
                method = user_input.split(':')[1].strip()
                kylebot.set_generation_method(method)
                continue
            elif not user_input:
                continue
            
            # Generate and display response
            print(f"\n🤖 KyleBot ({kylebot.generation_method}): ", end="")
            response = kylebot.generate_response(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Thanks for chatting with KyleBot!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again or type 'quit' to exit.")

if __name__ == "__main__":
    # Run tests first
    test_generation_methods()
    
    # Then start chat
    print("\n" + "=" * 50)
    chat_with_kylebot() 