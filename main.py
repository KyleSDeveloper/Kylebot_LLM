import logging
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Configure logging
logging.basicConfig(
    filename='kylebot_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load BART
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
# Placeholder: Load fine-tuned model later
# model.load_state_dict(torch.load('kylebot_bart_model.pt'))

def generate_response(user_input):
    try:
        # Prepare prompt with your chill, sarcastic vibe
        prompt = f"Kylebot, a sarcastic and witty AI assistant like Grok, responds to '{user_input}' in a chill, humorous tone:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=150,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "Yo, something broke. Gimme a sec to fix my circuits."

# Kylebot loop
print("Kylebot v0.1.0 - Your Sarcastic Cosmic Wingman (Powered by BART)")
while True:
    user_input = input("Yo, what’s the vibe? ")
    if user_input.lower() in ["quit", "exit", "bye", "stop"]:
        print("Later, my dude! Catch ya in the multiverse.")
        break
    response = generate_response(user_input)
    print(response)