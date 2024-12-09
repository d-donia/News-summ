def generate_summary(input, model, tokenizer):

    input_ids = tokenizer(
        input,
        padding='max_length',
        truncation=True,
        max_length=1024,
        return_tensors='pt'
    )

    print(f"Tokenized input IDs: {input_ids['input_ids']}")
    print(f"Input ID size: {input_ids['input_ids'].size()}")

    tokenized_output = model.generate(input_ids=input_ids['input_ids'], max_length=5000, length_penalty=1.5, num_beams=4)

    output = tokenizer.decode(tokenized_output[0], skip_special_tokens=True)

    return output