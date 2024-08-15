# src/utils.py

import logging
from requests.exceptions import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential
from scrapegraphai.graphs import SmartScraperGraph

# Set up logging
logging.basicConfig(filename='./logs/scraper.log', level=logging.ERROR)

def read_file_safe(filepath):
    """Read a file safely and handle any exceptions."""
    try:
        with open(filepath, 'r') as file:
            return file.readlines()
    except FileNotFoundError as e:
        logging.error(f"File not found: {filepath}")
        raise e
    except Exception as e:
        logging.error(f"Error reading file: {filepath}")
        raise e

def process_source(web_driver, url, max_len_html=200000):
    """Parse HTML content safely and handle any exceptions."""
    try:
        web_driver.get(url)
        html_content = web_driver.page_source
        if len(html_content) > max_len_html:
            html_content = html_content[:max_len_html]
        return html_content
    except Exception as e:
        logging.error("Error parsing HTML content")
        raise e

def call_llm_safe(prompt, html_content, config):
    """Call the LLM API safely and handle any exceptions."""
    try:
        return run_graph(prompt, html_content, config)
    except HTTPError as e:
        logging.error(f"HTTP error occurred: {e}")
        raise e
    except Exception as e:
        logging.error("Error in LLM API call")
        raise e

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_graph(prompt, url, config):
    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=config
    )
    return smart_scraper_graph.run()
