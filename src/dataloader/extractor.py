import os
import json
import requests
from typing import List, Dict, Any

class DataExtractor:
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def _get_story_points_field_id(self, base_url: str, auth: tuple = None) -> str:
        """
        Dynamically queries Jira's schema to find the custom field ID for 'Story Points'.
        """
        url = f"{base_url}/rest/api/2/field"
        headers = {"Accept": "application/json"}
        
        response = requests.get(url, headers=headers, auth=auth)
        if response.status_code == 200:
            fields = response.json()
            for field in fields:
                if field.get("name", "").lower() == "story points":
                    field_id = field.get("id")
                    print(f"[*] Dynamic mapping resolved: 'Story Points' -> '{field_id}'")
                    return field_id
                    
        print("[!] Warning: Could not dynamically resolve 'Story Points' field. Falling back to heuristic.")
        return None
    
    def _save(self, dataset: List[Dict[str, Any]]):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)
        print(f"[+] Successfully extracted {len(dataset)} stories.")
        print(f"[+] Dataset saved to {self.output_path}")

    def extract_jira(
        self, 
        project_key: str, 
        domain: str, 
        email: str = None, 
        api_token: str = None
    ) -> List[Dict[str, Any]]:
        print(f"[*] Extracting data from Jira project: {project_key} at {domain}")
        
        # Clean domain formatting
        domain = domain.rstrip("/")
        if not domain.startswith("http://") and not domain.startswith("https://"):
            base_url = f"https://{domain}"
        else:
            base_url = domain

        auth = (email, api_token) if email and api_token else None

        # 1. Dynamically resolve the Story Points custom field ID
        sp_field_id = self._get_story_points_field_id(base_url, auth)

        # 2. Query the issues
        jql = f"project = {project_key} AND 'Story Points' is not EMPTY"
        url = f"{base_url}/rest/api/2/search?jql={jql}&maxResults=100"
        
        headers = {"Accept": "application/json"}
        dataset = []

        response = requests.get(url, headers=headers, auth=auth)
        if response.status_code != 200:
            print(f"[!] Jira API Error ({response.status_code}): {response.text}")
            return dataset

        issues = response.json().get("issues", [])
        for issue in issues:
            fields = issue.get("fields", {})
            title = fields.get("summary", "")
            description = fields.get("description", "") or ""
            
            story = f"{title}\n\n{description}".strip()
            
            points = None
            
            # 3. Extract points using the dynamically mapped ID
            if sp_field_id and fields.get(sp_field_id) is not None:
                points = float(fields.get(sp_field_id))
            else:
                # Fallback: Guess the first numeric custom field if mapping failed
                for key, value in fields.items():
                    if "customfield" in key and isinstance(value, (int, float)):
                        points = float(value)
                        break

            if points is not None and story:
                dataset.append({"story": story, "points": points})

        self._save(dataset)
        return dataset

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
    

def run_extraction_pipeline(args):
    """Wrapper function to route CLI arguments to the extractor."""
    extractor = DataExtractor(output_path=args.out)
    
    if args.source == "github":
        token = os.getenv("GITHUB_TOKEN", getattr(args, "token", None))
        extractor.extract_github(repo=args.target, token=token)
        
    elif args.source == "jira":
        token = os.getenv("JIRA_API_TOKEN", getattr(args, "token", None))
        email = os.getenv("JIRA_EMAIL", getattr(args, "email", None))
        
        # ONLY require domain now. Email and token are strictly optional for public Jiras.
        if not args.domain:
            print("[!] Error: Jira extraction requires --domain.")
            return
            
        extractor.extract_jira(
            project_key=args.target, 
            domain=args.domain, 
            email=email, 
            api_token=token
        )