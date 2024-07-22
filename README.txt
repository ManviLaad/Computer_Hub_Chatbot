# PDF QA Chatbot

## Overview
This project is a Flask-based web application that uses BERT for question answering from a PDF document.

## How to Run the Code

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd PDF_QA_Chatbot
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv new_venv
    new_venv\Scripts\activate  # On Windows
    source new_venv/bin/activate  # On macOS/Linux
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download NLTK data**:
    ```sh
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
    ```

### Running the Application
1. **Start the Flask server**:
    ```sh
    python app.py
    ```

2. **Open your web browser and go to** `http://127.0.0.1:5000`.

## File Structure
- `app.py`: Main Flask application.
- `templates/index.html`: Frontend HTML file.
- `static/styles.css`: CSS styles.
- `static/script.js`: JavaScript functions.

## Troubleshooting
- Ensure all required packages are installed.
- Check the console for any error messages.

