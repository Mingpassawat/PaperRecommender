import requests
import time
import pandas as pd

# Replace with your API key
API_KEY = 'bdd76b3b043c7fba85551166fefa6721'
AFFILIATION_ID = '60072158'  # Chulalongkorn University's Scopus Affiliation ID (replace if needed)

# Base URL for Scopus Search API
BASE_URL = 'https://api.elsevier.com/content/search/scopus'

# Headers for the API request
headers = {
    'X-ELS-APIKey': API_KEY,
    'Accept': 'application/json'
}

# Function to fetch papers with pagination
def fetch_papers(affiliation_id, start=0, count=1000):
    results = []
    while start < count:
        params = {
            'query': f'AF-ID({affiliation_id})',  # Query by affiliation ID
            'start': start,  # Starting index for pagination
            'count': 25  # Number of results per request (max is 25 for Scopus API)
        }
        response = requests.get(BASE_URL, headers=headers, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            entries = data.get('search-results', {}).get('entry', [])
            results.extend(entries)
            
            # Break if no more results
            if len(entries) < 25:
                break
            
            # Increment the starting index for the next page
            start += 25
            time.sleep(1)  # Avoid hitting API rate limits
        else:
            print(f"Error: {response.status_code} - {response.text}")
            break
    
    return results

# Fetch papers
papers = fetch_papers(AFFILIATION_ID, count=1000)

# Convert results to DataFrame for analysis
if papers:
    df = pd.DataFrame(papers)
    # Save to CSV
    df.to_csv('chulalongkorn_papers.csv', index=False)
    print("Fetched and saved papers to 'chulalongkorn_papers.csv'")
else:
    print("No papers found.")