# Chatbot Transformer API

This project is a transformer-based chatbot API built using `DistilBERT`. It is trained on a customer service dataset and can classify user intents based on text input.

## Features
- **Intent Recognition:** Classifies customer queries into predefined intents.
- **Transformer Model:** Uses `DistilBERT` for efficient text classification.
- **FastAPI Backend:** Provides an endpoint for chatbot interaction.
- **Preprocessing & Training:** Includes scripts for data cleaning, tokenization, and model fine-tuning.
- **Model Evaluation:** Reports accuracy and classification metrics.

## Folder Structure
```
Chatbot_Project/
│── chatbot_app.py       # FastAPI application
│── model_training.ipynb # Notebook for model training
│── test_api.ipynb       # API testing notebook
│── requirements.txt     # Dependencies
│── label_mapping.json   # Mapping of intent labels
│── README.md            # Project documentation
│── datasets/            # Contains training & testing datasets
```

## Setup Instructions
### 1. Clone the repository
```sh
git clone <repo-url>
cd Chatbot_Project
```

### 2. Create a virtual environment & install dependencies
```sh
python -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model (Optional, if retraining is needed)
Run `model_training.ipynb` to fine-tune the chatbot model.

### 4. Start the API server
```sh
uvicorn chatbot_app:app --host 0.0.0.0 --port 8000
```

### 5. Test the API
Use a tool like Postman or run `test_api.ipynb` to send test requests.

## API Usage
**Endpoint:** `POST /chat`

**Request Body:**
```json
{
  "text": "How can I reset my password?"
}
```

**Response:**
```json
{
  "response": "Detected intent: Reset_Password"
}
```

## License
This project is open-source and free to use under the MIT license.

