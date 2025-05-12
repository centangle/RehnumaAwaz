from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def setup_deepseek():
    # Initialize the model and tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"  # Using the smaller 1.3B version
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name    )
    
    return model, tokenizer

def generate_support_ticket(model, tokenizer):
    prompt = """Generate a realistic customer support ticket with the following format:
Name: [Full Name]
Email: [Email Address]
Phone: [Phone Number]
Date: [Date of Incident]
Issue: [Detailed issue description]
"""
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_length=300,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate multiple tickets
def generate_dataset(num_tickets=1):
    model, tokenizer = setup_deepseek()
    tickets = []
    
    for i in range(num_tickets):
        try:
            ticket = generate_support_ticket(model, tokenizer)
            tickets.append(ticket)
            print(f"Generated ticket {i+1}/{num_tickets}")
        except Exception as e:
            print(f"Error generating ticket {i+1}: {str(e)}")
    
    return tickets


# Generate tickets
tickets = generate_dataset(10)

# Save the generated data
with open('support_tickets.txt', 'w') as f:
    f.write('\n---\n'.join(tickets))