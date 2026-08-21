import os
import json
import requests
from typing import List, Dict, Any

class DataExtractor:
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def extract_github(self, repo: str, token: str = None) -> List[Dict[str, Any]]:
        """
        Extracts issues from a GitHub repository.
        Requires issues to have labels indicating story points (e.g., "size: 5" or "points: 3").
        """
        print(f"[*] Extracting data from GitHub repository: {repo}")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"
        
        url = f"https://api.github.com/repos/{repo}/issues?state=all&per_page=100"
        dataset = []

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"[!] GitHub API Error: {response.text}")
            return dataset

        issues = response.json()
        for issue in issues:
            # Skip pull requests
            if "pull_request" in issue:
                continue
                
            story = f"{issue.get('title', '')}\n\n{issue.get('body', '')}".strip()
            
            # Simple heuristic: Look for a label that parses to a float (e.g., "3", "5.0", "size: 3")
            points = None
            for label in issue.get("labels", []):
                name = label.get("name", "").lower()
                # Example: parses labels like "size: 5" or just "5"
                try:
                    points = float(name.split(":")[-1].strip())
                    break
                except ValueError:
                    continue

            if points is not None and story:
                dataset.append({"story": story, "points": points})

        self._save(dataset)
        return dataset

    def extract_jira(self, project_key: str, domain: str, email: str, api_token: str) -> List[Dict[str, Any]]:
        """
        Extracts user stories and story points from a Jira project.
        """
        print(f"[*] Extracting data from Jira project: {project_key} at {domain}")
        # JQL query to get issues with story points assigned
        jql = f"project = {project_key} AND 'Story Points' is not EMPTY"
        url = f"https://{domain}/rest/api/3/search?jql={jql}&maxResults=100"
        
        auth = (email, api_token)
        headers = {"Accept": "application/json"}
        dataset = []

        response = requests.get(url, headers=headers, auth=auth)
        if response.status_code != 200:
            print(f"[!] Jira API Error: {response.text}")
            return dataset

        issues = response.json().get("issues", [])
        for issue in issues:
            fields = issue.get("fields", {})
            title = fields.get("summary", "")
            
            # Jira stores descriptions in a rich text Atlassian Document Format (ADF) in v3 API
            # For simplicity, we are grabbing the raw text if available, or just title.
            desc_obj = fields.get("description", {})
            desc_text = str(desc_obj) if desc_obj else "" 
            
            story = f"{title}\n\n{desc_text}".strip()
            
            # Custom field mapping for Story Points varies by Jira instance (usually customfield_10016 or similar)
            # You may need to adjust the exact field ID for your specific Jira instance.
            points = None
            for key, value in fields.items():
                if "customfield" in key and isinstance(value, (int, float)):
                    points = float(value) # Assuming the first float custom field is story points

            if points is not None and story:
                dataset.append({"story": story, "points": points})

        self._save(dataset)
        return dataset

    def _save(self, dataset: List[Dict[str, Any]]):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)
        print(f"[+] Successfully extracted {len(dataset)} stories.")
        print(f"[+] Dataset saved to {self.output_path}")


def run_extraction_pipeline(args):
    """Wrapper function to route CLI arguments to the extractor."""
    extractor = DataExtractor(output_path=args.out)
    
    if args.source == "github":
        token = os.getenv("GITHUB_TOKEN", args.token)
        extractor.extract_github(repo=args.target, token=token)
        
    elif args.source == "jira":
        token = os.getenv("JIRA_API_TOKEN", args.token)
        email = os.getenv("JIRA_EMAIL", args.email)
        if not args.domain or not email or not token:
            print("[!] Error: Jira extraction requires --domain, JIRA_EMAIL, and JIRA_API_TOKEN.")
            return
        extractor.extract_jira(project_key=args.target, domain=args.domain, email=email, api_token=token)