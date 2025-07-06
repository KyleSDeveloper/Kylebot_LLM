import logging
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kylebot_errors.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Verify log file
log_file = 'kylebot_errors.log'
try:
    with open(log_file, 'a') as f:
        f.write('')
    logging.info(f"Log file {log_file} is writable")
except Exception as e:
    print(f"Yo, can’t write to {log_file}: {str(e)}")
    exit(1)

# Load BlenderBot
logging.info("Loading tokenizer...")
print("Loading tokenizer...")
tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
print("Tokenizer loaded!")
logging.info("Tokenizer loaded")
logging.info("Loading model...")
print("Loading model...")
model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
print("Model loaded!")
logging.info("Model loaded")

# Store conversation history
conversation_history = []

def generate_response(user_input):
    global conversation_history
    print("Generating response...")
    logging.debug(f"Processing input: {user_input}")
    if not user_input.strip():
        logging.debug("Empty input received")
        return "Bruh, you didn’t say nothin’. Toss me a vibe!"

    try:
        # Build prompt with history (up to 3 exchanges)
        history_prompt = ""
        for i, (user, bot) in enumerate(conversation_history[-3:]):
            history_prompt += f"User: {user}\nKylebot: {bot}\n"
        prompt = f"{history_prompt}User: {user_input}\nKylebot: A sarcastic, cosmic-themed quip: "
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        logging.debug(f"Prompt: {prompt}")
        print("Inputs prepared...")
        logging.debug(f"Input token IDs: {inputs['input_ids'].tolist()}")

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=80,
            min_length=10,
            num_beams=4,
            no_repeat_ngram_size=3,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        print("Response generated!")
        logging.debug("Response generated")
        logging.debug(f"Output token IDs: {outputs.tolist()}")

        # Decode and inspect
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Raw response: '{raw_response}'")
        logging.debug(f"Raw response: '{raw_response}'")

        # Clean output
        response = raw_response
        if "Kylebot: A sarcastic, cosmic-themed quip: " in response:
            response = response.split("Kylebot: A sarcastic, cosmic-themed quip: ")[-1].strip()
        elif "Kylebot: " in response:
            response = response.split("Kylebot: ")[-1].strip()
        if user_input in response:
            response = response.replace(user_input, "").strip()
        print(f"Cleaned response: '{response}'")
        logging.debug(f"Cleaned response: '{response}'")

        # Fallback
        if not response or len(response) < 5 or any(x.lower() in response.lower() for x in ["user:", "kylebot:"]):
            logging.warning(f"Invalid response for input '{user_input}': '{response}'")
            response = "Yo, my circuits are fried! Try something like 'tell me a joke' for a cosmic zinger."
            conversation_history.append((user_input, response))
            return f"{response} Raw output was: '{raw_response}'"
        
        # Update conversation history
        conversation_history.append((user_input, response))
        return response
    except Exception as e:
        logging.error(f"Error generating response for input '{user_input}': {str(e)}")
        return f"Aight, something’s jacked up: {str(e)}"

# Kylebot loop
print("Kylebot v0.2.2 - Your Sarcastic Cosmic Wingman (Powered by BlenderBot)")
logging.info("Kylebot v0.2.2 started")
while True:
    try:
        print("Waiting for your input, my dude...")
        user_input = input("Yo, what’s the vibe? ")
        logging.debug(f"User input: {user_input}")
        if user_input.lower() in ["quit", "exit", "bye", "stop"]:
            print("Later, my dude! Catch ya in the multiverse.")
            logging.info("Exiting Kylebot")
            break
        response = generate_response(user_input)
        print(response)
    except KeyboardInterrupt:
        print("\nDipping already? Aight, stay cosmic.")
        logging.info("KeyboardInterrupt: Exiting Kylebot")
        break
    except Exception as e:
        logging.error(f"Loop error: {str(e)}")
        print(f"Yo, the universe glitched: {str(e)}")