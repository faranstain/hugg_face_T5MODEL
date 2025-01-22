from transformers import T5ForConditionalGeneration, T5Tokenizer


model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def generate_text(input_text, task="summarize"):
    
    input_text = f"{task}: {input_text}"

    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
   
    outputs = model.generate(inputs['input_ids'], max_length=50)
  
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

input_text = input("Please enter a sentence: ")


output_text = generate_text(input_text, task="summarize") 

print("Generated Text:")
print(output_text)