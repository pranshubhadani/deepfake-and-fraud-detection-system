import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

DATASET_DIR = r'Datasets\SMS Datasets'
MODEL_FILE = 'sms_scam_model.pkl'

def load_and_prepare_data():
    """Loads all CSV datasets and prepares a unified dataframe with both 'spam' and 'ham'."""
    all_data = []

    if not os.path.exists(DATASET_DIR):
        print(f"Directory {DATASET_DIR} not found.")
        return pd.DataFrame(columns=['text', 'label'])

    for filename in os.listdir(DATASET_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(DATASET_DIR, filename)
            try:
                # Read CSV, skipping bad lines if any exist
                df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
                
                # Standardize column names
                df.columns = [str(col).lower().strip() for col in df.columns]
                
                # Extract text and label based on known dataset headers
                if 'msg' in df.columns and 'label' in df.columns:
                    temp_df = df[['msg', 'label']].rename(columns={'msg': 'text'})
                elif 'v1' in df.columns and 'v2' in df.columns:
                    temp_df = df[['v2', 'v1']].rename(columns={'v2': 'text', 'v1': 'label'})
                elif 'target' in df.columns and 'text' in df.columns:
                    temp_df = df[['text', 'target']].rename(columns={'target': 'label'})
                else:
                    print(f"Skipping {filename}: Unknown column format.")
                    continue
                
                # Normalize labels to just 'spam' and 'ham'
                temp_df['label'] = temp_df['label'].astype(str).str.strip().str.lower()
                temp_df = temp_df[temp_df['label'].isin(['spam', 'ham'])]
                
                # Drop empty text rows
                temp_df = temp_df.dropna(subset=['text'])
                
                all_data.append(temp_df)
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not all_data:
        return pd.DataFrame(columns=['text', 'label'])

    # Combine all datasets and remove duplicates
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['text'])
    
    return combined_df

def train_model():
    """Trains the ML model and saves it to disk."""
    print("Loading and preparing datasets...")
    df = load_and_prepare_data()
    
    if df.empty:
        print("No data found to train the model.")
        return None

    # Count how many spam and ham messages we have
    spam_count = len(df[df['label'] == 'spam'])
    ham_count = len(df[df['label'] == 'ham'])
    print(f"Training model on {len(df)} unique messages ({spam_count} scams, {ham_count} safe)...")
    
    # Create an ML pipeline: Text -> TF-IDF Math Vectors -> Naive Bayes Classifier
    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    
    # Train the model!
    model.fit(df['text'], df['label'])
    
    # Save the trained model to a file so we don't have to retrain every time
    joblib.dump(model, MODEL_FILE)
    print(f"Model trained and saved as '{MODEL_FILE}'.")
    
    return model

def get_model():
    """Loads the model from disk, or trains it if it doesn't exist."""
    if os.path.exists(MODEL_FILE):
        print(f"Loading pre-trained model from {MODEL_FILE}...")
        return joblib.load(MODEL_FILE)
    else:
        print("Pre-trained model not found. Initiating training sequence...")
        return train_model()

def check_sms(message, model):
    """
    Scans the message using the ML model.
    Returns (is_scam_boolean, confidence_percentage)
    """
    # Predict the label ('spam' or 'ham')
    prediction = model.predict([message])[0]
    
    # Get the probability (confidence score)
    probabilities = model.predict_proba([message])[0]
    
    # Find which index corresponds to 'spam'
    spam_index = list(model.classes_).index('spam')
    confidence = probabilities[spam_index] * 100

    is_scam = prediction == 'spam'
    
    return is_scam, round(confidence, 2)

if __name__ == '__main__':
    # Initialize the model (loads from file or trains from scratch)
    model = get_model()
    
    if model:
        print("\n" + "="*40)
        print(" 🤖 ML SMS Scam Detector Ready ")
        print("="*40)
        while True:
            test_message = input("\nEnter an SMS message to check (or type 'exit' to quit): ")
            
            if test_message.strip().lower() == 'exit':
                break
                
            if not test_message.strip():
                continue
                
            is_scam, confidence = check_sms(test_message, model)
            
            if is_scam:
                print(f"🚨 WARNING: This is likely a SCAM.")
                print(f"📊 Confidence: {confidence}%")
            else:
                safe_confidence = round(100 - confidence, 2)
                print(f"✅ SAFE: This message appears to be normal.")
                print(f"📊 Confidence: {safe_confidence}%")
