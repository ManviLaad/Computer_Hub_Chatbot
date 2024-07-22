import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def generate_response(question, context):
    # Preprocess question and context
    question = question.strip()
    context = context.strip()
    
    # Tokenize and encode inputs
    max_length = 512
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=max_length)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        # Run model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1
    
    answer_tokens = input_ids[0][start_index+1:end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Truncate long answers
    answer_max_length = 300
    if len(answer) > answer_max_length:
        answer = answer[:answer_max_length] + '...'
    
    return answer
