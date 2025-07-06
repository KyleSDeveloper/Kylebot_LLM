# KyleBot: A GPT-2 Chatbot for Learning

This project demonstrates how to build a generative AI chatbot using GPT-2. It includes educational comments to help you understand what's happening.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Gradio web interface (Recommended):**
   ```bash
   # Option A: Use the launcher script (easiest)
   ./start_kylebot.sh
   
   # Option B: Manual start
   source kylebot_env/bin/activate
   python kylebot_gradio.py
   ```
   Then open your browser to: `http://localhost:7860`

3. **Or run the command-line version:**
   ```bash
   python kylebot_fixed.py
   ```

4. **Or use the Jupyter notebook:**
   ```bash
   jupyter notebook kylebot.ipynb
   ```

## 🤖 What You'll Learn

- **Model Loading & Setup**: How to load pre-trained language models
- **Text Generation Strategies**: 
  - Greedy Decoding (always picks most likely word)
  - Sampling (creative, uses temperature and top-k)
  - Beam Search (explores multiple possibilities)
- **Conversational Interface**: Building an interactive chat loop
- **Parameter Tuning**: Understanding how different parameters affect responses

## 🎮 How to Use

### Web Interface (Gradio):
- **Modern UI**: Beautiful, responsive web interface
- **Real-time chat**: Type messages and get instant responses
- **Parameter controls**: Adjust temperature, top-k, and max tokens with sliders
- **Method switching**: Easily switch between generation methods
- **Clear chat**: Start fresh conversations with one click
- **Mobile friendly**: Works on phones and tablets

### Command Line Interface:
- `quit` - Exit the chat
- `method: [greedy/sampling/beam]` - Change generation method
- `history` - See conversation history
- `help` - Show this help message
- `test` - Run generation method tests

### Generation Methods:

1. **Greedy Decoding** (`method: greedy`)
   - Always picks the most likely next word
   - Fast and predictable
   - Good for simple, factual responses

2. **Sampling** (`method: sampling`)
   - Uses temperature and top-k parameters
   - More creative and diverse responses
   - Temperature: 0.1 (focused) to 1.5 (creative)
   - Top-k: Limits word choices to top k most likely

3. **Beam Search** (`method: beam`)
   - Explores multiple possible sequences
   - Balanced quality and coherence
   - Uses num_beams parameter (more beams = potentially better quality)

## 🔧 Key Parameters

- **Temperature**: Controls randomness (0.1 = focused, 1.5 = creative)
- **Top-k**: Limits word choices to top k most likely
- **Max Length**: Controls response length
- **Num Beams**: Number of parallel searches in beam search
- **No Repeat N-gram Size**: Prevents repetition of phrases
- **Repetition Penalty**: Reduces repetitive text

## 🌟 Features

### Web Interface (Gradio)
- **Modern Design**: Clean, professional interface with Soft theme
- **Real-time Chat**: Instant responses with typing indicators
- **Parameter Controls**: Interactive sliders for temperature, top-k, and max tokens
- **Method Selection**: Dropdown to switch between generation methods
- **Chat History**: Persistent conversation memory
- **Mobile Responsive**: Works great on all devices
- **Network Access**: Can be accessed from other devices on your network

### Command Line Interface
- **Interactive Chat**: Terminal-based conversation
- **Method Testing**: Built-in tests for all generation methods
- **History Management**: View and clear conversation history
- **Parameter Experimentation**: Easy parameter adjustment

## 🐛 Issues Fixed

The original notebook had a parameter passing issue where the `generate_response_greedy` function didn't accept `**kwargs`. The fixed versions resolve this by:

1. Adding `**kwargs` to all generation functions
2. Properly passing parameters through to the model.generate() calls
3. Removing debug print statements for cleaner output
4. Creating a beautiful web interface with Gradio

## 💡 Pro Tips

- Start with sampling (temperature 0.7-0.9) for most use cases
- Use beam search for factual or technical responses
- Greedy decoding is good for simple, predictable tasks
- Always clean and format your responses
- Keep conversation history for context

## 🚀 Next Steps to Explore

1. **Fine-tuning**: Train the model on your own data
2. **Different Models**: Try GPT-3, BERT, or other models
3. **Web Interface**: Build a web app for your chatbot
4. **Memory**: Add long-term conversation memory
5. **Personality**: Customize the bot's responses
6. **Multi-turn**: Handle complex conversations

Happy learning! 🎉

## 📚 Project Structure

```
llm/
├── kylebot_learning.ipynb    # Main learning notebook
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore              # Git ignore file
```

## 🧠 Key Concepts Covered

### Text Generation Methods

1. **Greedy Decoding**
   - Always picks the most likely next word
   - Fast but can be repetitive
   - Good for simple, predictable tasks

2. **Sampling**
   - Randomly selects from likely words
   - More creative and diverse
   - Controlled by temperature and top-k parameters

3. **Beam Search**
   - Explores multiple possible sequences
   - Balanced quality and coherence
   - Slower but often better results

### Important Parameters

- **Temperature**: Controls randomness (0.1 = focused, 1.5 = creative)
- **Top-k**: Limits word choices to top k most likely
- **Max Length**: Controls response length
- **Num Beams**: Number of parallel searches in beam search

## 🎓 Learning Path

1. **Start Simple**: Use the basic chat interface
2. **Experiment**: Try different parameters and methods
3. **Understand**: Read the educational comments
4. **Customize**: Modify the code to add features
5. **Build**: Create your own chatbot variations

## 📖 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Text Generation Strategies](https://huggingface.co/blog/how-to-generate)

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new features
- Improve the documentation
- Share your experiments
- Ask questions

Happy learning! 🎉

---

**Note**: This project uses GPT-2, which is a powerful but older model. For production use, consider newer models like GPT-3, GPT-4, or open-source alternatives.