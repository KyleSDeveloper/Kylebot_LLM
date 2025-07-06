# KyleBot: A GPT-2 Chatbot with Gradio Web Interface
# Beautiful, modern web interface for your AI chatbot

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import re
import gradio as gr
import time

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
            return f"✅ Generation method set to: {method}"
        else:
            return f"❌ Invalid method. Choose from: {valid_methods}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "🗑️ Conversation history cleared!"

# Create our chatbot instance
kylebot = KyleBot()

def chat_with_bot(message, history, method, temperature, top_k, max_tokens):
    """Main chat function for Gradio interface"""
    if not message.strip():
        return "", history
    
    # Update bot's generation method
    kylebot.set_generation_method(method)
    
    # Generate response based on method
    if method == "greedy":
        response = kylebot.generate_response(message, method="greedy", max_new_tokens=max_tokens)
    elif method == "sampling":
        response = kylebot.generate_response(message, method="sampling", 
                                           temperature=temperature, top_k=top_k, 
                                           max_new_tokens=max_tokens)
    elif method == "beam":
        response = kylebot.generate_response(message, method="beam", 
                                           num_beams=5, max_new_tokens=max_tokens)
    else:
        response = kylebot.generate_response(message, method="sampling", 
                                           temperature=temperature, top_k=top_k, 
                                           max_new_tokens=max_tokens)
    
    # Add to history
    history.append((message, response))
    return "", history

def clear_chat():
    """Clear the chat interface"""
    kylebot.clear_history()
    return []

def get_method_info(method):
    """Get information about the selected generation method"""
    info = {
        "greedy": "🤖 **Greedy Decoding**: Always picks the most likely next word. Fast and predictable, good for factual responses.",
        "sampling": "🎲 **Sampling**: Uses temperature and top-k for creative, diverse responses. More random and creative.",
        "beam": "🔍 **Beam Search**: Explores multiple possible sequences. Balanced quality and coherence."
    }
    return info.get(method, "Select a generation method to see its description.")

# Create the Gradio interface
with gr.Blocks(
    title="KyleBot - AI Chatbot",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🤖 KyleBot - AI Chatbot
    
    Welcome to KyleBot! This is a GPT-2 powered chatbot that demonstrates different text generation strategies.
    
    ## 🎮 How to Use
    - Type your message and press Enter or click Send
    - Choose your preferred generation method
    - Adjust parameters to control the response style
    - Use the Clear button to start a new conversation
    
    ## 🔧 Generation Methods
    - **Greedy**: Fast, predictable responses
    - **Sampling**: Creative, diverse responses  
    - **Beam Search**: Balanced quality and coherence
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat with KyleBot",
                height=500,
                show_label=True,
                container=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
        
        with gr.Column(scale=1):
            # Controls panel
            gr.Markdown("### ⚙️ Settings")
            
            method = gr.Dropdown(
                choices=["sampling", "greedy", "beam"],
                value="sampling",
                label="Generation Method",
                info="Choose how KyleBot generates responses"
            )
            
            method_info = gr.Markdown(get_method_info("sampling"))
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.8,
                step=0.1,
                label="Temperature",
                info="Controls randomness (higher = more creative)"
            )
            
            top_k = gr.Slider(
                minimum=10,
                maximum=100,
                value=50,
                step=10,
                label="Top-K",
                info="Limits word choices to top K most likely"
            )
            
            max_tokens = gr.Slider(
                minimum=20,
                maximum=200,
                value=100,
                step=10,
                label="Max Tokens",
                info="Maximum length of response"
            )
            
            gr.Markdown("### 📊 Model Info")
            gr.Markdown(f"""
            - **Model**: GPT-2
            - **Parameters**: {model.num_parameters():,}
            - **Status**: ✅ Ready
            """)
    
    # Event handlers
    def handle_message(message, history, method_val, temp, top_k_val, max_tok):
        return chat_with_bot(message, history, method_val, temp, top_k_val, max_tok)
    
    def update_method_info(method_val):
        return get_method_info(method_val)
    
    # Connect events
    send_btn.click(
        handle_message,
        inputs=[msg, chatbot, method, temperature, top_k, max_tokens],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        handle_message,
        inputs=[msg, chatbot, method, temperature, top_k, max_tokens],
        outputs=[msg, chatbot]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot]
    )
    
    method.change(
        update_method_info,
        inputs=[method],
        outputs=[method_info]
    )
    
    # Update method info when method changes
    method.change(
        update_method_info,
        inputs=[method],
        outputs=[method_info]
    )

if __name__ == "__main__":
    print("🚀 Starting KyleBot Gradio Interface...")
    print("📱 Open your browser to the URL shown below")
    print("💡 You can also access it from other devices on your network")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        show_error=True,
        quiet=False
    ) 