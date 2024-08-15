# src/scraper.py

import os
import time
import pandas as pd
from scrapegraphai.graphs import SmartScraperGraph
import nest_asyncio
from utils import read_file_safe, process_source, call_llm_safe
from selenium import webdriver
from dotenv import load_dotenv

# Apply nest_asyncio to resolve any issues with asyncio event loop
nest_asyncio.apply()

def main():
    """Main function to run the scraper tool."""
    
    # Load environment variables
    load_dotenv(dotenv_path='./config/config.env')

    # Set API keys and configuration
    groqcloud_api_key = os.getenv('GROQCLOUD_API_KEY')
    use_cloud_llm = os.getenv('USE_CLOUD_LLM', 'True') == 'True'
    embedding_model = os.getenv('EMBEDDING_MODEL')
    cloud_model = os.getenv('CLOUD_MODEL')
    local_model = os.getenv('LOCAL_MODEL')
    programs_file = os.getenv('PROGRAMS_LIST_PATH')


    # Set prompt
    prompt = """List the following as specified.
    Program title: name of educational program;
    Institution: name of educational institution;
    Credential: educational credential offered;
    Attendance: Full-time, Part-time, Online;
    Tuition: program fee or cost;
    Scholarships: available or not;
    Internship: internship, co-op provided or not;
    Accommodation: yes or no;
    Job_assistance: yes or no for career guidance;
    Alumni_network: yes or no;
    Application_deadline: format ddmmyyyy;
    Start_date: format ddmmyyyy"""

    if use_cloud_llm:
        llm_config = {
            "llm": {
                "model": cloud_model,
                "api_key": groqcloud_api_key,
                "temperature": 0
            },
            "embeddings": {
                "model": embedding_model,
            },
            "verbose": False,
        }
    else:
        llm_config = {
            "llm": {
                "model": local_model,
                "temperature": 0,
                "format": "json",
                "base_url": "http://localhost:11434",
            },
            "embeddings": {
                "model": embedding_model,
                "base_url": "http://localhost:11434",
            },
            "verbose": False,
        }

    # Load program data
    programs_df = pd.read_excel(programs_file, sheet_name="Programs")

    # Initialize WebDriver

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.headless = True
    chrome_options.add_argument('--log-level=3')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--silent')
    chrome_options.add_argument('--disable-dev-shm-usage')

    web_driver = webdriver.Chrome(options=chrome_options)

    # Process each URL to obtain the HTML source
    programs_df['HTML Source'] = programs_df['URL'].apply(lambda url: process_source(web_driver, url))

    web_driver.quit()

    # Prepare DataFrame for results
    url_df = pd.DataFrame(programs_df[['URL', 'HTML Source']])
    results = []

    for index, row in url_df.iterrows():
        
        try:
            result = call_llm_safe(prompt, row['HTML Source'], llm_config)
            results.append(result)
            print(f"Processed {index+1} out of {len(url_df)} URLs")
        except Exception as e:
            print(f"Error processing URL {row['URL']}: {e}")
            results.append(None)

        time.sleep(5)

    # Compile results into DataFrame
    results_df = pd.DataFrame(results, columns=['Program title', 'Institution', 'Credential', 'Attendance', 'Tuition', 'Scholarships', 'Internship', 'Accommodation', 'Job_assistance', 'Alumni_network', 'Application_deadline', 'Start_date'])
    results_df = results_df.fillna('')

    # Save the results to files
    results_df.to_csv("./data/output/results.csv", index=False)
    combined_df = pd.concat([programs_df, results_df], axis=1)
    combined_df.to_excel("./data/output/programs_scraped.xlsx", index=False)

if __name__ == "__main__":
    main()
