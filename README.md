# Gemma 3 Fine-tuning for Indian Penal Code

This project fine-tunes Google's Gemma 3 4B model to create a specialized legal assistant for the Indian Penal Code (IPC). The implementation uses Unsloth for efficient fine-tuning on Google Colab with T4 GPU.

## Project Overview
- **Base Model**: unsloth/gemma-3-4b-it-unsloth-bnb-4bit
- **Training Method**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **Dataset**: NahOR102/Indian-IPC-Laws from Hugging Face
- **Purpose**: Create an AI legal assistant specialized in Indian criminal law
- **Hardware**: Optimized for Google Colab T4 GPU environment

## Dataset

This project uses the [Indian-IPC-Laws](https://huggingface.co/datasets/NahOR102/Indian-IPC-Laws) dataset from Hugging Face, which contains structured information about various sections of the Indian Penal Code. The dataset is formatted as a collection of conversations between users and legal assistants, making it suitable for fine-tuning a conversational AI model.

## Requirements

```
unsloth
bitsandbytes
accelerate
xformers==0.0.29.post3
peft
trl
triton
cut_cross_entropy
unsloth_zoo
sentencepiece
protobuf
datasets
huggingface_hub
hf_transfer
transformers==4.49.0
torch>=2.0.0
```

## Setup and Usage

1. **Clone the Repository**:
   ```bash
   https://github.com/rohanmalik102003/gemma-3-indian-penal-code-finetuning.git
   cd gemma-3-indian-penal-code-finetuning
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Environment**:
   - This code is optimized for Google Colab with T4 GPU
   - Clone the dataset:
     ```bash
     git clone https://huggingface.co/datasets/NahOR102/Indian-IPC-Laws
     ```

4. **Run Training**:
   ```bash
   python train.py
   ```

5. **Test the Model**:
   ```bash
   python test_model.py
   ```

## Model Features

The fine-tuned model can:
- Answer questions about specific IPC sections and their provisions
- Explain legal concepts in Indian criminal law (e.g., mens rea, culpable homicide vs. murder)
- Provide information about punishments for various offenses
- Clarify legal defenses and procedural aspects of Indian criminal law

## Memory Usage

The code includes memory tracking to optimize for Google Colab's T4 GPU:
- Training uses 4-bit quantization for efficient memory usage
- Peak memory usage statistics are provided during training
- The model is optimized to run within T4 GPU memory constraints

## Acknowledgements

- [Unsloth](https://github.com/unsloth/unsloth) for the optimized fine-tuning library
- Google for the [Gemma 3](https://blog.google/technology/developers/gemma-open-models/) model
- [NahOR102](https://huggingface.co/NahOR102) for the Indian-IPC-Laws dataset

## License

This project is released under the [MIT License](LICENSE).

## Disclaimer

This model is intended for educational and research purposes only. It should not be used as a substitute for professional legal advice. Always consult a qualified legal professional for specific legal matters.
