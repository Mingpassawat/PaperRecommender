import requests
import csv
import time

# API endpoint and your parameters
url = "https://api.elsevier.com/content/search/scopus"
params = {
    'count': 25,  # Reduced to 25 results per request to avoid exceeding limits
    'start': 0,
    'apiKey': 'bdd76b3b043c7fba85551166fefa6721',  # Use your API key
}

# Define the subjects you're interested in
subjects = [
    'AGRI', 'ARTS', 'BIOC', 'BUSI', 'CENG', 'CHEM', 'COMP', 'DECI', 'DENT', 
    'EART', 'ECON', 'ENER', 'ENGI', 'ENVI', 'HEAL', 'IMMU', 'MATE', 'MATH', 
    'MEDI', 'MULT', 'NEUR', 'NURS', 'PHAR', 'PHYS', 'PSYC', 'SOCI', 'VETE'
]

# Function to fetch papers for a specific subject with pagination
def fetch_papers(subject, target_papers=56, retries=3):
    papers = []
    start = 0
    fetched_papers = 0  # Track number of fetched papers
    retry_count = 0

    while fetched_papers < target_papers:
        if retry_count >= retries:
            print(f"Maximum retries reached for subject: {subject}. Skipping.")
            break
        
        params['query'] = f'TITLE-ABS-KEY({subject})'  # Flexible query format
        params['start'] = start
        params['count'] = 25  # Fetch 25 papers per request
        
        try:
            response = requests.get(url, params=params, timeout=10)  # Add timeout to prevent hanging
        except requests.exceptions.Timeout:
            print(f"Timeout occurred while fetching data for {subject}. Retrying...")
            retry_count += 1
            time.sleep(5)  # Wait for 5 seconds before retrying
            continue
        
        # Check if response is not successful
        if response.status_code != 200:
            print(f"Failed to fetch data for {subject}, Status code: {response.status_code}, Response: {response.text}")
            break
        
        data = response.json()
        entries = data.get('search-results', {}).get('entry', [])
        
        # Stop further requests if no results are returned
        if not entries:
            print(f"No more results found for subject: {subject}. Stopping.")
            break

        # Process each entry in the response
        for entry in entries:
            title = entry.get('dc:title', 'Unknown')
            keywords = entry.get('authkeywords', [])
            cited_by = entry.get('citedby-count', '0')
            
            # Check if 'subject-area' is available, otherwise set 'Unknown'
            subject_area = entry.get('subject-area', ['Unknown'])[0] if entry.get('subject-area') else 'Unknown'
            keywords_str = "; ".join(keywords) if keywords else "None"
            
            papers.append({
                'title': title,
                'query_subject': subject,  # Add the current query subject
                'subjects': subject_area,
                'keywords': keywords_str,
                'cited_by': cited_by
            })
            fetched_papers += 1  # Increment the number of papers fetched
        
        # Stop if we have fetched the desired number of papers
        if fetched_papers >= target_papers:
            break
        
        start += 25  # Update the start parameter for the next batch
        retry_count = 0  # Reset retries if successful

        print(f"Fetched {fetched_papers}/{target_papers} papers for subject: {subject}.")
    
    return papers

# Function to write papers to CSV
def write_papers_to_csv(papers, filename='papers.csv'):
    header = ['title', 'query_subject', 'subjects', 'keywords', 'cited_by']
    
    # Write to CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for paper in papers:
            writer.writerow(paper)

# Main function to fetch and write papers for all subjects
def fetch_and_save_data():
    all_papers = []
    target_papers_per_subject = 56  # Set to 56 papers per subject

    for subject in subjects:
        print(f"Fetching up to {target_papers_per_subject} papers for subject: {subject}")
        papers = fetch_papers(subject, target_papers_per_subject)
        all_papers.extend(papers)  # Add the papers to the list of all papers

    # Write the collected papers to CSV
    write_papers_to_csv(all_papers)

# Start the data fetching and saving process
fetch_and_save_data()