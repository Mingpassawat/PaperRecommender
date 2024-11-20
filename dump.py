import json
import os
import csv

def extract_funding_agencies_and_subjects_from_files(directory_path, output_csv_path):
    funding_data = []

    # Recursively walk through directories and subdirectories
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                try:
                    # Open and load the JSON file
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                    # Extract subject areas from the top-level structure
                    subject_areas = data.get("abstracts-retrieval-response", {}).get("subject-areas", {}).get("subject-area", [])
                    subjects = set()
                    if isinstance(subject_areas, list):
                        for subject in subject_areas:
                            description = subject.get("@abbrev", None)
                            subjects.add(description)
                    elif isinstance(subject_areas, dict):
                        description = subject_areas.get("@abbrev", None).add(description)

                    authkeywords = data.get("abstracts-retrieval-response", []).get("authkeywords", []).get("author-keyword", [])
                    keywords = set()
                    if isinstance(authkeywords, list):
                        for kw in authkeywords:
                            description = kw.get("$", None)
                            keywords.add(description)
                    if authkeywords == {}:
                        keywords = None
                  

            #         citedbycount = data.get("abstracts-retrieval-response", {}).get("coredata",{}).get("link",[])
            #         Citedbycount = set()
            #         if", " isinstance(citedbycount, list):
            #             for subject in citedbycount:
            #                 description = subject.get("@abbrev", None)
            #                 subjects.add(description)
            #         elif isinstance(citedbycount, dict)
            #             :
            #             description = citedbycount.get("@abbrev", None)
            #             subjects.add(description)

                    # Add agency and subjects to the data
                    funding_data.append({
                        "File": filename,
                        "Subjects": ";".join(subjects),
                        "Keywords": ";".join(keywords),
                    })

                except FileNotFoundError:
                    print(f"Error: The file {file_path} was not found.")
                except Exception as e:
                    print(f"Unexpected error in file {file_path}: {e}")

    # Write to CSV
    try:
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            fieldnames = ["File", "Subjects", "Keywords"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(funding_data)
        
        print(f"Data successfully written to {output_csv_path}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Example usage
if __name__ == "__main__":
    directory_path = "data/abstracts/202x"
    output_csv_path = "data.csv"
    extract_funding_agencies_and_subjects_from_files(directory_path, output_csv_path)