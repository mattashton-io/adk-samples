from google.cloud import logging
import os
from dotenv import load_dotenv

def fetch_deployment_logs(project_id, build_id=None):
    client = logging.Client(project=project_id)
    
    # Filter for Cloud Build and Cloud Run deployment events
    filter_str = (
        f'resource.type="cloud_run_revision" OR '
        f'resource.type="build" '
        f'severity>=ERROR'
    )
    
    # Retrieve logs
    entries = client.list_entries(filter_=filter_str, order_by=logging.DESCENDING, page_size=50)
    
    log_data = []
    for entry in entries:
        log_data.append({
            "timestamp": entry.timestamp,
            "text_payload": entry.payload,
            "resource": entry.resource.type
        })
    return log_data

if __name__ == "__main__":
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "next-26-adk-demo")
    print(f"Fetching logs for project: {project_id}")
    logs = fetch_deployment_logs(project_id)
    for log in logs:
        print(f"--- {log['timestamp']} [{log['resource']}] ---")
        print(log['text_payload'])
        print("-" * 40)