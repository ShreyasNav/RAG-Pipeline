import argparse
import requests
from bs4 import BeautifulSoup

# Function to get the URL of the closest Wikipedia article using SerpAPI
def get_wikipedia_url(query, serpapi_key):
    url = "https://serpapi.com/search"
    params = {
        "q": query + " site:wikipedia.org",  # Focus search on Wikipedia
        "api_key": serpapi_key,              # Your SerpAPI key
        "num": 1                             # Retrieve only the first result
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error for bad responses
    
    data = response.json()
    
    # Extract the first Wikipedia URL from the organic search results
    for result in data.get("organic_results", []):
        if "wikipedia.org" in result.get("link", ""):
            return result["link"]
    
    raise Exception("No Wikipedia article found for the given query.")

# Function to scrape text from the Wikipedia article
def scrape_wikipedia_article(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract paragraphs from the main content
    content = soup.find_all("p")
    article_text = "\n".join(p.get_text() for p in content)

    return article_text

# Function to save the article text to a file
def save_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)

# Main function
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Fetch closest Wikipedia article.")
    parser.add_argument("query", type=str, help="The topic to search for.")
    parser.add_argument("serpapi_key", type=str, help="Your SerpAPI key.")
    parser.add_argument("--output", type=str, default="output.txt", help="Output filename.")
    args = parser.parse_args()

    try:
        print("Searching for the closest Wikipedia article...")
        wikipedia_url = get_wikipedia_url(args.query, args.serpapi_key)
        print(f"Found Wikipedia article: {wikipedia_url}")

        print("Scraping the article content...")
        article_text = scrape_wikipedia_article(wikipedia_url)

        print(f"Saving the article to {args.output}...")
        save_to_file(article_text, args.output)
        print("Article saved successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


