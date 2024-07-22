import json
import logging

def extract_text_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logging.info(f"Loaded JSON data: {data}")

            text = ''
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        text += value + ' '
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str):
                                text += value + ' '

            if not text:
                logging.warning("No text found in the JSON file.")
            else:
                logging.info(f"Extracted text: {text[:500]}...")

        return text
    except Exception as e:
        logging.error(f"Error extracting text from JSON: {e}")
        return ""

# Example usage
if __name__ == '__main__':
    json_path = r'C:\Users\IT\Desktop\Computer_hub\Corpus (4).json'
    extracted_text = extract_text_from_json(json_path)
    print(extracted_text)
