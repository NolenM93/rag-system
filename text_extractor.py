"""
text_extractor.py - Extract text from web pages or PDF files
"""

import requests
from bs4 import BeautifulSoup


def scrape_web_page(url):
    """
    Fetches a web page, extracts all paragraph text from the main content,
    and writes it to Selected_Document.txt.
    
    Args:
        url (str): The URL to fetch
        
    Returns:
        str: The extracted text, or None if failed
    """
    try:
        # Fetch the page with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check status code
        if response.status_code != 200:
            print(f"Failed to fetch page. Status code: {response.status_code}")
            return None
            
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract all <p> tags (paragraphs)
        paragraphs = soup.find_all('p')
        
        # Join their text with blank lines
        text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # Write to file
        with open('Selected_Document.txt', 'w', encoding='utf-8') as f:
            f.write(text)
            
        print(f"Successfully extracted {len(paragraphs)} paragraphs from {url}")
        print(f"Text saved to Selected_Document.txt ({len(text)} characters)")
        
        return text
        
    except Exception as e:
        print(f"Error fetching or parsing page: {e}")
        return None


def main():
    """
    Main function to run the web scraper with a hardcoded URL.
    Change this URL to scrape different pages.
    """
    # Hardcoded URL - change this to your desired Wikipedia article or webpage
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    print(f"Fetching content from: {url}")
    scrape_web_page(url)


if __name__ == '__main__':
    main()
