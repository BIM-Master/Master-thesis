import os
import pandas as pd
import numpy as np
import re
import pdfplumber
import nltk
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import openai
import itertools
import logging

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set your OpenAI API key here
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# File paths
LM_LEXICON_PATH = '/Users/albinbergstrom/Desktop/THESIS CODE/Loughran-McDonald_MasterDictionary_1993-2024.csv'
PDF_PATH_FINANCE = '/Users/albinbergstrom/Desktop/THESIS CODE/Recent Developments in Finance (2024–2025).pdf'
PDF_PATH_LEGAL = '/Users/albinbergstrom/Desktop/THESIS CODE/Global Legal Developments (Mid-2024 – Mid-2025).pdf'
CSV_OUTPUT_PATH = 'Final_dataset_thesis.csv'

# Model configuration
MODEL_NAME = 'all-mpnet-base-v2'
BATCH_SIZE = 100

# --- NLTK SETUP ---
def download_nltk_resources():
    """Downloads necessary NLTK resources if not already present."""
    resources_to_download = ['punkt', 'stopwords']
    for resource in resources_to_download:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            logging.info(f"NLTK resource '{resource}' already downloaded.")
        except LookupError:
            logging.info(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource, quiet=True)

download_nltk_resources()

# --- 1. PROMPT GENERATION ---
def generate_prompts():
    """Generate all combinations of prompts based on the independent variables"""
    logging.info("Generating all prompt combinations...")
    
    character_tuning = {
        "None": "",
        "Peer": "You are my peer at the company, offering advice based on our similar position",
        "Mentor": "You are my mentor at this company, offering advice from your more experienced perspective"
    }
    
    prompter_tuning = {
        "None": "",
        "Entry-level": "I am an entry-level employee working at a firm within the [Topic] department",
        "Middle-manager": "I am a middle manager working at a firm within the [Topic] department",
        "C-suite": "I am a C-suite executive working at a firm within the [Topic] department"
    }
    
    topics = ["Finance", "Legal"]
    prompter_ratings = list(range(1, 11)) + [None]
    
    logging.info(f"Generating {len(character_tuning)} × {len(prompter_tuning)} × {len(topics)} × {len(prompter_ratings)} = {len(character_tuning) * len(prompter_tuning) * len(topics) * len(prompter_ratings)} total prompts")
    
    prompts = []
    for char_name, char_text in character_tuning.items():
        for prompt_name, prompt_text in prompter_tuning.items():
            for topic in topics:
                for rating in prompter_ratings:
                    components = []
                    
                    if char_text:
                        components.append(f"{char_text};")
                    
                    if prompt_text:
                        components.append(f"{prompt_text.replace('[Topic]', topic)}:")
                    
                    components.append(f"What are some interesting things I should know about recent developments/findings within {topic}?")
                    
                    if rating is not None:
                        components.append(f"If I would rate my pre-existing knowledge on this topic on a scale of 1-10, I'd rate myself a {rating}.")
                    
                    components.append("Thank you.")
                    
                    final_prompt = " ".join(components)
                    
                    prompts.append({
                        "prompt": final_prompt,
                        "character_tuning": char_name,
                        "prompter_tuning": prompt_name,
                        "topic": topic,
                        "prompter_rating": rating
                    })
    
    df_prompts = pd.DataFrame(prompts)
    logging.info(f"Generated {len(df_prompts)} prompts successfully")
    return df_prompts

# --- 2. LLM RESPONSE COLLECTION ---
def get_gpt4_response(prompt, temperature=0.7, model="gpt-4"):
    """Get response from GPT-4 API with error handling"""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content, response.usage.total_tokens, True, None
    except Exception as e:
        logging.error(f"Error getting GPT-4 response: {e}")
        return None, None, False, str(e)

def collect_responses(df_prompts, sample_size=None):
    """Collect responses from GPT-4 for all prompts"""
    logging.info("Starting LLM response collection...")
    
    if sample_size:
        df_prompts = df_prompts.head(sample_size)
        logging.info(f"Using sample size of {sample_size} prompts")
    
    responses = []
    total_prompts = len(df_prompts)
    
    for idx, row in df_prompts.iterrows():
        logging.info(f"Processing prompt {idx + 1}/{total_prompts}")
        
        resp, tokens, success, error = get_gpt4_response(row["prompt"])
        
        responses.append({
            **row,
            "response": resp,
            "tokens": tokens,
            "success": success,
            "error": error
        })
        
        # Save progress every 10 prompts
        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(responses)
            temp_df.to_csv("temp_responses.csv", index=False)
            logging.info(f"Saved progress: {idx + 1}/{total_prompts} responses collected")
    
    df_responses = pd.DataFrame(responses)
    df_responses.to_csv("prompt_responses.csv", index=False)
    logging.info(f"Response collection complete. Saved {len(df_responses)} responses")
    return df_responses

# --- 3. LEXICON LOADING & PREPROCESSING ---
def load_and_filter_lm_lexicon(file_path=LM_LEXICON_PATH):
    """Loads the Loughran-McDonald lexicon and removes stopwords"""
    logging.info("Loading Loughran-McDonald lexicon...")
    
    try:
        lm_df = pd.read_csv(file_path, keep_default_na=False, na_values=[''])
        
        if 'Word' not in lm_df.columns:
            logging.error("Error: 'Word' column not found in LM lexicon CSV.")
            return set()
        
        all_lm_words = set(lm_df['Word'].astype(str).str.lower().tolist())
        logging.info(f"Original LM lexicon size: {len(all_lm_words)} words")
        
        stop_words_set = set(nltk.corpus.stopwords.words('english'))
        lm_words_no_stopwords = {word for word in all_lm_words if word not in stop_words_set}
        
        logging.info(f"LM lexicon size after stopword removal: {len(lm_words_no_stopwords)} words")
        return lm_words_no_stopwords
        
    except FileNotFoundError:
        logging.error(f"Error: LM Lexicon file not found at {file_path}")
        return set()
    except Exception as e:
        logging.error(f"Error processing LM lexicon: {e}")
        return set()

def preprocess_text(text):
    """Basic text preprocessing: lowercase, tokenize, remove punctuation"""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    return tokens

def count_lexicon_terms(tokens, lexicon_set):
    """Counts occurrences of terms from the lexicon_set in the given tokens"""
    if not tokens or not lexicon_set:
        return 0, []
    
    term_counts = Counter(tokens)
    total_lexicon_matches = 0
    matched_words_list = []
    
    for term in lexicon_set:
        if term in term_counts:
            total_lexicon_matches += term_counts[term]
            matched_words_list.extend([term] * term_counts[term])
    
    return total_lexicon_matches, sorted(list(set(matched_words_list)))

def min_max_scale_series(series_to_scale, scale_min=1, scale_max=10):
    """Applies Min-Max scaling to a pandas Series to a custom range [scale_min, scale_max]"""
    if series_to_scale.isna().all():
        return series_to_scale
    
    min_val = series_to_scale.min()
    max_val = series_to_scale.max()
    
    if pd.isna(min_val) or pd.isna(max_val):
        return series_to_scale
    
    if max_val == min_val:
        return pd.Series(np.where(series_to_scale.notna(), scale_min, np.nan), index=series_to_scale.index)
    
    scaled_series = scale_min + (series_to_scale - min_val) * (scale_max - scale_min) / (max_val - min_val)
    return scaled_series

def perform_lexicon_analysis(df_responses):
    """Perform lexicon-based analysis on responses"""
    logging.info("Starting lexicon analysis...")
    
    # Load lexicon
    lm_lexicon = load_and_filter_lm_lexicon()
    if not lm_lexicon:
        logging.error("Failed to load lexicon. Exiting lexicon analysis.")
        return df_responses
    
    # Preprocess responses
    df_responses['processed_response_tokens'] = df_responses['response'].apply(preprocess_text)
    
    # Count lexicon terms
    temp_lm_results = df_responses['processed_response_tokens'].apply(
        lambda tokens: count_lexicon_terms(tokens, lm_lexicon)
    )
    
    df_responses['lm_filtered_finance_term_count'] = temp_lm_results.apply(lambda x: x[0])
    df_responses['lm_matched_words'] = temp_lm_results.apply(lambda x: ', '.join(x[1]))
    
    # Calculate response length
    df_responses['response_length'] = df_responses['response'].fillna('').astype(str).map(len)
    
    # Calculate jargon density
    df_responses['lm_jargon_density'] = df_responses.apply(
        lambda row: row['lm_filtered_finance_term_count'] / row['response_length'] 
        if row['response_length'] > 0 else 0,
        axis=1
    )
    
    # Scale scores
    df_responses['lm_jargon_score_scaled'] = min_max_scale_series(df_responses['lm_jargon_density'], 1, 10)
    df_responses['response_length_scaled'] = min_max_scale_series(df_responses['response_length'], 1, 10)
    
    logging.info("Lexicon analysis complete")
    return df_responses

# --- 4. SEMANTIC SIMILARITY ---
def extract_text_from_pdf(pdf_path, doc_type="Document"):
    """Extracts all text from a PDF file"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logging.info(f"Successfully extracted text from {doc_type}. Length: {len(text)} characters.")
        return text
    except Exception as e:
        logging.error(f"Error reading PDF for {doc_type}: {e}")
        return None

def calculate_similarity_bert(text1, text2, model):
    """Calculates cosine similarity between two texts using a sentence transformer model"""
    if not text1 or not isinstance(text1, str) or not text1.strip():
        return None
    if not text2 or not isinstance(text2, str) or not text2.strip():
        return None
    
    try:
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_scores.item()
    except Exception as e:
        logging.error(f"Error calculating BERT-based similarity: {e}")
        return None

def perform_semantic_similarity_analysis(df_responses):
    """Perform semantic similarity analysis"""
    logging.info("Starting semantic similarity analysis...")
    
    # Load sentence transformer model
    logging.info(f"Loading sentence transformer model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        logging.info(f"Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model '{MODEL_NAME}': {e}")
        return df_responses
    
    # Extract benchmark texts
    logging.info("Extracting benchmark texts...")
    benchmark_text_finance = extract_text_from_pdf(PDF_PATH_FINANCE, "Finance Benchmark")
    benchmark_text_legal = extract_text_from_pdf(PDF_PATH_LEGAL, "Legal Benchmark")
    
    if not benchmark_text_finance or not benchmark_text_legal:
        logging.error("Could not extract text from benchmark PDFs. Exiting similarity analysis.")
        return df_responses
    
    # Initialize similarity columns
    df_responses['finance_benchmark_similarity'] = np.nan
    df_responses['legal_benchmark_similarity'] = np.nan
    
    # Calculate similarities
    total_responses = len(df_responses)
    for idx, row in df_responses.iterrows():
        if (idx + 1) % 10 == 0:
            logging.info(f"Processing similarity for response {idx + 1}/{total_responses}")
        
        llm_response = row['response']
        topic = row['topic']
        
        if pd.isna(llm_response) or not isinstance(llm_response, str) or not llm_response.strip():
            continue
        
        if topic == "Finance":
            sim = calculate_similarity_bert(llm_response, benchmark_text_finance, model)
            df_responses.at[idx, 'finance_benchmark_similarity'] = sim
        elif topic == "Legal":
            sim = calculate_similarity_bert(llm_response, benchmark_text_legal, model)
            df_responses.at[idx, 'legal_benchmark_similarity'] = sim
    
    # Scale similarity scores
    logging.info("Scaling similarity scores...")
    
    finance_scores = df_responses.loc[df_responses['topic'] == 'Finance', 'finance_benchmark_similarity']
    legal_scores = df_responses.loc[df_responses['topic'] == 'Legal', 'legal_benchmark_similarity']
    
    df_responses.loc[df_responses['topic'] == 'Finance', 'finance_similarity_scaled'] = min_max_scale_series(finance_scores, 1, 10)
    df_responses.loc[df_responses['topic'] == 'Legal', 'legal_similarity_scaled'] = min_max_scale_series(legal_scores, 1, 10)
    
    logging.info("Semantic similarity analysis complete")
    return df_responses

# --- MAIN PIPELINE ---
def main():
    """Main pipeline that runs all steps from prompt generation to final dataset"""
    logging.info("Starting complete pipeline...")
    
    # Step 1: Generate prompts
    df_prompts = generate_prompts()
    
    # Step 2: Collect LLM responses (commented out for safety - uncomment to run)
    # df_responses = collect_responses(df_prompts, sample_size=10)  # Use sample_size for testing
    
    # For now, load existing responses if available
    try:
        df_responses = pd.read_csv("prompt_responses.csv")
        logging.info(f"Loaded existing responses from prompt_responses.csv: {len(df_responses)} responses")
    except FileNotFoundError:
        logging.info("No existing responses found. Please run collect_responses() first or uncomment the line above.")
        logging.info("For testing, you can use: df_responses = collect_responses(df_prompts, sample_size=10)")
        return
    
    # Step 3: Perform lexicon analysis
    df_responses = perform_lexicon_analysis(df_responses)
    
    # Step 4: Perform semantic similarity analysis
    df_responses = perform_semantic_similarity_analysis(df_responses)
    
    # Step 5: Remove failed generations
    if 'success' in df_responses.columns:
        initial_len = len(df_responses)
        df_responses = df_responses[df_responses['success'] != False].reset_index(drop=True)
        removed = initial_len - len(df_responses)
        logging.info(f"Removed {removed} rows where success == False")
    
    # Step 6: Save final dataset
    df_responses.to_csv(CSV_OUTPUT_PATH, index=False)
    logging.info(f"Final dataset saved to {CSV_OUTPUT_PATH}")
    logging.info(f"Final dataset contains {len(df_responses)} rows")
    
    # Display sample of results
    logging.info("Sample of final dataset:")
    cols_to_show = ['prompt', 'topic', 'response', 'lm_jargon_score_scaled', 'response_length_scaled']
    if 'finance_similarity_scaled' in df_responses.columns:
        cols_to_show.append('finance_similarity_scaled')
    if 'legal_similarity_scaled' in df_responses.columns:
        cols_to_show.append('legal_similarity_scaled')
    
    cols_to_show = [col for col in cols_to_show if col in df_responses.columns]
    print(df_responses[cols_to_show].head())

if __name__ == "__main__":
    main() 