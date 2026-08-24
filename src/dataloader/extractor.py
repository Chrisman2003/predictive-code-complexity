import os
import json
import requests
import random
import sys
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
    
    def _save(self, dataset: List[Dict[str, Any]], filepath: str = None):
        save_path = filepath or self.output_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)
            
        print(f"[+] Saved {len(dataset)} records to {save_path}")    
        

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
        url = f"{base_url}/rest/api/2/search?jql={jql}&maxResults=5000"
        
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
                dataset.append({"key": issue.get("key"), "story": story, "points": points})

        print(f"[+] Successfully extracted {len(dataset)} stories.")
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
                dataset.append({"key": issue.get("key"),"story": story, "points": points})

        print(f"[+] Successfully extracted {len(dataset)} stories.")
        return dataset
    

def run_extraction_pipeline(args):
    """Wrapper function to route CLI arguments to the extractor."""
    extractor = DataExtractor(output_path=args.out)
    dataset = []
    
    # ==========================================
    # EXTRACTION LOGIC
    # ==========================================
    if args.source == "github":
        token = os.getenv("GITHUB_TOKEN", getattr(args, "token", None))
        dataset = extractor.extract_github(repo=args.target, token=token)
        
    elif args.source == "jira":
        token = os.getenv("JIRA_API_TOKEN", getattr(args, "token", None))
        email = os.getenv("JIRA_EMAIL", getattr(args, "email", None))
        
        # ONLY require domain now. Email and token are strictly optional for public Jiras.
        if not args.domain:
            print("[!] Error: Jira extraction requires --domain.")
            return
            
        dataset = extractor.extract_jira(
            project_key=args.target, 
            domain=args.domain, 
            email=email, 
            api_token=token
        )
        
    if not dataset:
        print(f"[!] No issues found for target '{args.target}' with the current filters.")
        print("[!] Aborting save/split operations.")
        return  # Stops the function completely
        
    # ==========================================
    # SPLITTING LOGIC
    # ==========================================
    # If the user passed --split (e.g., --split 60 20 20) AND we got data back
    if getattr(args, "split", None):
        train_pct, val_pct, test_pct = args.split
        
        if sum(args.split) != 100:
            print(f"[!] Error: Split percentages {args.split} do not equal 100.")
            sys.exit(1)

        print(f"[*] Shuffling and partitioning dataset into {train_pct}% Train, {val_pct}% Val, {test_pct}% Test...")
        
        random.seed(42)
        random.shuffle(dataset)
        
        total = len(dataset)
        train_end = int(total * (train_pct / 100.0))
        val_end = train_end + int(total * (val_pct / 100.0))
        
        splits = {
            "train": dataset[:train_end],
            "val": dataset[train_end:val_end],
            "test": dataset[val_end:]
        }
        
        # Build the dynamic path: data/processed/Mesos/
        target_dir = args.target.replace("/", "_")
        base_dir = os.path.join("data", "processed", target_dir)
        os.makedirs(base_dir, exist_ok=True)
        
        target_prefix = target_dir.lower()

        for split_name, split_data in splits.items():
            # Generate the dynamic path for this specific split
            filepath = os.path.join(base_dir, f"{target_prefix}_dataset_{split_name}.json")
            extractor._save(split_data, filepath=filepath)