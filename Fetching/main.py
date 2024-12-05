import requests
import pandas as pd
import time

# Configuration
API_KEY = "bdd76b3b043c7fba85551166fefa6721"  # Replace with your Elsevier Scopus API Key
BASE_URL = "https://api.elsevier.com/content/search/scopus"
QUERY = "machine learning"  # Replace with your desired query
RESULTS_PER_PAGE = 25       # Scopus API allows up to 25 results per request
TOTAL_RESULTS = 1000        # Total number of papers to fetch
OUTPUT_FILE = "scopus_papers.csv"

# Function to fetch papers from Scopus API
def fetch_scopus_data(query, start, api_key):
    params = {
        "query": query,
        "count": RESULTS_PER_PAGE,
        "start": start,
        "view": "STANDARD",  # Use STANDARD for broader compatibility
    }
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }
    response = requests.get(BASE_URL, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Main script to fetch and save data
def main():
    all_results = []
    for start in range(0, TOTAL_RESULTS, RESULTS_PER_PAGE):
        print(f"Fetching results {start + 1} to {start + RESULTS_PER_PAGE}...")
        data = fetch_scopus_data(QUERY, start, API_KEY)
        if data and "search-results" in data:
            entries = data["search-results"].get("entry", [])
            all_results.extend(entries)
            print(entries)
        else:
            print("No more results or an error occurred.")
            break
        time.sleep(2)  # Add delay to avoid rate-limiting

    # Extract relevant fields and save to a CSV
    print(f"Total results fetched: {len(all_results)}")
    papers = []
    for entry in all_results:
        
        papers.append({
            "Title": entry.get("dc:title"),
            "Authors": entry.get("dc:creator"),
            "Publication Name": entry.get("prism:publicationName"),
            "Publication Year": entry.get("prism:coverDate"),
            "DOI": entry.get("prism:doi"),
            "Link": entry.get("link", [{}])[0].get("@href")
        })

    # Save to CSV
    df = pd.DataFrame(papers)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()