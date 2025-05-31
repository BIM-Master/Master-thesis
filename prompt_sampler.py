import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
import requests  # Use requests for HTTP calls
from prompt_generator import generate_prompts
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import time
import random
import json

# Load environment variables
load_dotenv()

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

class PromptSampler:
    def __init__(self, sample_size=100, db_path="prompt_responses.csv"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.sample_size = sample_size
        self.db_path = db_path
        self.pickle_path = self.db_path.replace('.csv', '.pkl')
        
    def generate_samples(self):
        """Generate random samples from all possible prompts"""
        all_prompts = generate_prompts()
        return random.choices(all_prompts, k=self.sample_size)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process_prompt(self, prompt_data: dict) -> dict:
        """Process a single prompt with retry logic using requests"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt_data["prompt"]}],
            "temperature": 0.7
        }
        
        try:
            print(f"\nSending prompt: {prompt_data['prompt'][:100]}...")
            response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            response_data = response.json()
            
            print(f"Received response of {response_data['usage']['total_tokens']} tokens")
            return {
                **prompt_data,
                "response": response_data['choices'][0]['message']['content'],
                "tokens": response_data['usage']['total_tokens'],
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request Error: {str(e)}")
            error_message = str(e)
            if e.response is not None:
                try:
                    error_details = e.response.json()
                    error_message += f" - {error_details.get('error', {}).get('message', '')}"
                except json.JSONDecodeError:
                    error_message += f" - {e.response.text[:100]}..." # Show start of non-JSON error response
            return {
                **prompt_data,
                "response": "",
                "tokens": 0,
                "success": False,
                "error": f"HTTP Error: {error_message}",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
            return {
                **prompt_data,
                "response": "",
                "tokens": 0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def process_samples(self):
        """Process all samples and save results, appending to existing CSV if present."""
        samples = self.generate_samples()
        print(f"\nProcessing {self.sample_size} new samples...")
        new_results_list = []
        pbar = tqdm(total=self.sample_size, desc="Processing new samples")
        
        for prompt_data in samples:
            result = self.process_prompt(prompt_data)
            new_results_list.append(result)
            pbar.update(1)
            time.sleep(1)  # Increased delay to be safe with direct API calls
        
        pbar.close()
        new_df = pd.DataFrame(new_results_list)
        
        # Define column order once
        column_order = [
            'prompt', 'response', 'character_tuning', 'prompter_tuning',
            'topic', 'prompter_rating', 'tokens', 'success', 'error',
            'timestamp'
        ]

        # Ensure new_df has all columns in the correct order, even if some are all NaN
        for col in column_order:
            if col not in new_df.columns:
                new_df[col] = None # Or appropriate default like 0 for tokens, False for success
        new_df = new_df[column_order]

        # Load existing data if CSV exists and append new results
        if os.path.exists(self.db_path):
            print(f"\nLoading existing data from {self.db_path}...")
            try:
                existing_df = pd.read_csv(self.db_path)
                # Ensure existing_df also has all columns in the correct order
                for col in column_order:
                    if col not in existing_df.columns:
                        existing_df[col] = None 
                existing_df = existing_df[column_order]
                
                print(f"Found {len(existing_df)} existing records.")
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"Appending {len(new_df)} new records.")
            except pd.errors.EmptyDataError:
                print(f"{self.db_path} is empty. Starting with new results.")
                combined_df = new_df
            except Exception as e:
                print(f"Error reading existing CSV {self.db_path}: {e}. Overwriting with new results.")
                combined_df = new_df # Fallback to overwrite if reading fails badly
        else:
            print("No existing CSV found. Creating new file.")
            combined_df = new_df
        
        print("\nSaving combined data...")
        combined_df.to_csv(self.db_path, index=False)
        combined_df.to_pickle(self.pickle_path) # Pickle file will always contain the full (combined) dataset
        
        print(f"\nResults saved. Total records in {self.db_path}: {len(combined_df)}")
        print(f"Pickle file updated at {self.pickle_path}")
        
        # --- Display summary for the combined_df ---
        df_to_summarize = combined_df # Use combined_df for summary
        print("\n--- Overall Summary ---")
        print(f"Total prompts processed (all runs): {len(df_to_summarize)}")
        successful_responses_all = df_to_summarize[df_to_summarize['success'] == True]
        failed_responses_all = df_to_summarize[df_to_summarize['success'] == False]

        print(f"Successful responses (all runs): {len(successful_responses_all)}")
        print(f"Failed responses (all runs): {len(failed_responses_all)}")
        
        avg_tokens = 0
        if not successful_responses_all.empty:
            avg_tokens = successful_responses_all['tokens'].mean()
        print(f"Average tokens per successful response (all runs): {avg_tokens:.2f}")
        
        print("\nPrompt distribution (all runs):")
        if not df_to_summarize.empty:
            print("\nCharacter Tuning distribution:")
            print(df_to_summarize['character_tuning'].value_counts(dropna=False))
            print("\nPrompter Tuning distribution:")
            print(df_to_summarize['prompter_tuning'].value_counts(dropna=False))
            print("\nTopic distribution:")
            print(df_to_summarize['topic'].value_counts(dropna=False))
            
            # Display first few of the NEWLY ADDED responses for quick check
            print("\n--- First Few NEWLY ADDED Responses ---")
            for i, row in new_df.head().iterrows(): # Iterate over new_df for this part
                print(f"\nPrompt {i+1} from this run (Success: {row['success']}):")
                print(f"Character Tuning: {row['character_tuning']}")
                print(f"Prompter Tuning: {row['prompter_tuning']}")
                print(f"Topic: {row['topic']}")
                print(f"Rating: {row['prompter_rating']}")
                print(f"Tokens: {row['tokens']}")
                if not row['success']:
                    print(f"Error: {row['error']}")
                print("-" * 80)
                print("Prompt:", row['prompt'])
                print("-" * 80)
                print("Response:", row['response'])
                print("=" * 80)
        else:
            print("No data to display.")
            
        return combined_df # Return the full combined DataFrame

def main():
    sampler = PromptSampler(sample_size=100) # Change sample_size to 2000 for full run
    df = sampler.process_samples()

if __name__ == "__main__":
    main()


    import pandas as pd

# Load the DataFrame from the pickle file (usually faster and preserves data types)
try:
    df = pd.read_pickle("prompt_responses.pkl")
    print("Loaded data from prompt_responses.pkl")
except FileNotFoundError:
    print("prompt_responses.pkl not found, trying CSV...")
    try:
        df = pd.read_csv("prompt_responses.csv")
        print("Loaded data from prompt_responses.csv")
    except FileNotFoundError:
        print("Neither prompt_responses.pkl nor prompt_responses.csv found.")
        df = pd.DataFrame() # Create an empty DataFrame if no file found

if not df.empty:
    print("\n--- Prompts and Responses ---")
    for index, row in df.iterrows():
        print(f"\n--- Entry {index + 1} ---")
        print(f"Prompt:\n{row['prompt']}\n")
        print(f"Response:\n{row['response']}\n")
        print("-" * 30)
else:
    print("No data to display.")

print(df)
