from conversion import extract_text_from_json
from app import split_text_into_chunks
from qa_system import generate_response

json_path = r'C:\Users\IT\Desktop\Computer_hub\Corpus (4).json'
text = extract_text_from_json(json_path)

# Example questions
questions = [
    "What makes Jessup Cellars wines special?",
    "Are dogs allowed at Jessup Cellars?",
    "Tell me about white wine?",
    "What does your white wine pair well with?",
    "What white wines do you have?",
    "What red wines is Jessup Cellars offering in 2024?",
    "Please tell me more about your consulting winemaker Rob Lloyd?"
]

# Split context into chunks
chunks = split_text_into_chunks(text)
# print("hello")
for question in questions:
    answers = []
    for chunk in chunks:
        response = generate_response(question, chunk)
        print('Iteration1')
        answers.append(response)
    combined_answer = " ".join(answers).strip()
    print(f"if chunks loop is not working then print this question: {question}")
    print(f"print answers Answer: {combined_answer}\n")
