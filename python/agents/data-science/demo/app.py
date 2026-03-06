import os
import logging
import asyncio
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

# Import the root agent from the data_science package
from data_science.agent import root_agent

# Load environment variables
load_dotenv("../.env")

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ADK services
session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

# Initialize the Runner
runner = Runner(
    app_name="DataScienceDemo",
    agent=root_agent,
    artifact_service=artifact_service,
    session_service=session_service,
)

# In-memory store for session IDs per user (simplified for demo)
user_sessions = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Helper to run async code in a sync Flask route
    return asyncio.run(_handle_chat(request.json))

async def _handle_chat(data):
    user_id = data.get("user_id", "demo_user")
    message = data.get("message")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Get or create session for the user
    if user_id not in user_sessions:
        session = await session_service.create_session(
            app_name="DataScienceDemo",
            user_id=user_id,
        )
        user_sessions[user_id] = session.id
    
    session_id = user_sessions[user_id]

    try:
        content = types.Content(role="user", parts=[types.Part(text=message)])
        events = []
        
        # Run the agent and collect events
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            events.append(event)

        # Extract the final response text
        last_event = events[-1]
        response_text = "".join(
            [part.text for part in last_event.content.parts if part.text]
        )

        # Check for generated artifacts (e.g., images/plots)
        artifacts = []
        session_artifacts = await artifact_service.list_artifacts(session_id=session_id)
        for artifact in session_artifacts:
            artifacts.append({
                "id": artifact.id,
                "name": artifact.name,
                "url": f"/artifacts/{artifact.id}"
            })

        return jsonify({
            "response": response_text,
            "artifacts": artifacts
        })

    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/artifacts/<artifact_id>")
def get_artifact(artifact_id):
    return asyncio.run(_handle_get_artifact(artifact_id))

async def _handle_get_artifact(artifact_id):
    try:
        artifact = await artifact_service.get_artifact(artifact_id)
        if not artifact:
            return "Artifact not found", 404
        
        content = await artifact.read()
        
        from flask import Response
        return Response(content, mimetype='image/png') 

    except Exception as e:
        logger.error(f"Error retrieving artifact: {e}")
        return str(e), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
