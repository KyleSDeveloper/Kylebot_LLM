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
print("Loading tokenizer...")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
print("Tokenizer loaded!")
print("Loading model...")
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
print("Model loaded!")

def generate_response(user_input):
    print("Generating response...")
    try:
        # Craft prompt to guide BART toward a response
        prompt = f"Kylebot, a sarcastic and witty AI assistant, responds to '{user_input}' with a chill, humorous answer:"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        print("Inputs prepared...")
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,  # Reduced for speed
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=2,
            decoder_start_token_id=tokenizer.bos_token_id  # Start with beginning of sentence
        )
        print("Response generated!")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        if not response or prompt in response:
            response = "Yo, I’m still figuring this out—try again, my dude!"
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