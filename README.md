# Llama 3.1 Fine-tuning with Unsloth

This project provides a comprehensive guide to fine-tuning and deploying the Meta-Llama-3.1-8B-Instruct large language model, leveraging the Unsloth library for optimized training and inference speeds. The process begins with the crucial steps of model and tokenizer initialization, ensuring the model is correctly set up and ready for fine-tuning. We then delve into Parameter-Efficient Fine-Tuning (PEFT) configuration, specifically focusing on LoRA (Low-Rank Adaptation) to efficiently adapt the pre-trained model to a specific task.

A key aspect of the project is data preparation, where we utilize a custom dataset, "FatumMaster/test_GOR", to tailor the model's learning. This involves loading, cleaning, and formatting the data to suit the model's input requirements. Model training is then conducted using the SFTTrainer from the `trl` (Transformer Reinforcement Learning) library, a powerful tool for supervised fine-tuning. After the training phase, the project demonstrates how to perform inference with the fine-tuned model, evaluating its performance on the target task.

Furthermore, the guide includes detailed steps for saving and pushing the fine-tuned LoRA model to the Hugging Face Hub, facilitating model sharing and collaboration. To ensure the model's accessibility and usability, we cover the process of loading and testing the saved model. Finally, for broader deployment options, the project provides instructions on optionally converting and pushing the model to GGUF (GPT Understandable Format), making it compatible with various inference engines. The Unsloth library accelerates the training and inference, making it a valuable tool for this process.

# SmolLM2 Custom Dataset Fine-tuning

This project focuses on the fine-tuning of a language model, specifically HuggingFaceTB/SmolLM2-135M-Instruct, using a custom dataset. The dataset can be either "FatumMaster/test_GOR" or a local file "gor_dataset.jsonl". This process allows adapting the pre-trained model to specific tasks or domains.

The initial step involves loading the dataset using appropriate tools and libraries, followed by preprocessing to clean, filter, and format the data into a suitable structure for model training. Text tokenization is then performed, transforming the textual data into numerical representations understandable by the language model.  A data collator is created, specifically designed for language modeling, which batches and prepares the data for efficient training.

The pre-trained base model (HuggingFaceTB/SmolLM2-135M-Instruct) is loaded from the Hugging Face Model Hub. Training arguments are carefully defined to control the training process.  These include essential parameters such as the output directory for saving the fine-tuned model, the number of training epochs determining the training duration, the batch size which influences the amount of data processed in each iteration, and the learning rate that governs the speed and stability of the training process.

The Trainer class, a central component of the Hugging Face Transformers library, is initialized using the loaded model, tokenized dataset, data collator, and defined training arguments. The fine-tuning process is then initiated, where the model iteratively learns from the dataset and adjusts its parameters to improve its performance. Finally, after the training is complete, the fine-tuned model is saved to the specified output directory. This saved model can then be used for downstream tasks.

# DistilGPT-2 Math Question Answering

This project details the fine-tuning of a DistilGPT-2 model specifically for math question answering, using the "HuggingFaceH4/MATH-500" dataset. The process starts by loading the dataset, which contains mathematical problems and their corresponding solutions.

A crucial step is preprocessing the dataset to format the questions and answers in a way that's suitable for the model. This might involve concatenating questions and answers with special tokens to delineate them or transforming the text into a specific format the model expects. Tokenization is then performed using the DistilGPT-2 tokenizer, converting the text into numerical tokens the model can process.

Next, a data collator is created for language modeling. This component is responsible for batching the tokenized data and preparing it for training. It typically handles tasks such as padding sequences to a uniform length. Training arguments are defined to configure the training process. Key arguments include the output directory where the fine-tuned model will be saved, and the batch size which determines how many examples are processed in each training step. Other common training arguments, not explicitly mentioned but often crucial, include the learning rate, number of epochs, and weight decay.

The Trainer is initialized with the DistilGPT-2 model, the processed dataset, the tokenizer, the data collator, and the defined training arguments. Finally, the fine-tuning process is initiated, where the model learns to answer math questions from the dataset. Once training is complete, the fine-tuned model is saved to the specified output directory for later use. This fine-tuned model can then be used to answer new math questions.
