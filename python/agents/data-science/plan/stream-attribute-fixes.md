# **Stream Attribute: Data Science Agent Fixes**

Use these prompts within the Antigravity environment to fix the AttributeError and align your deployment scripts with the Vertex AI Reasoning Engine SDK.

## **1\. Fix Root Cause: Schema Registration (CRITICAL)**

**Goal:** Fix the Operation schemas: \[\] issue. If the schema is empty, the .stream() method will not exist on the deployed object.

"The agent is deploying with Operation schemas: \[\], causing an AttributeError later. Please update deployment/deploy.py to correctly diagnose this:

1. In deployment/deploy.py, within the create function, change the diagnostic line from root\_agent.operation\_schemas() to adk\_app.operation\_schemas(). The AdkApp wrapper is what generates the schemas for Vertex AI, not the LlmAgent itself.  
2. Add a check: if not adk\_app.operation\_schemas(): logger.error('ADK App generated an empty schema. Deployment will fail to execute.').  
3. Ensure data\_science/agent.py exports root\_agent as a fully initialized instance. If the schemas are still empty, we may need to wrap the LlmAgent call in a simple class that explicitly defines a stream method with type hints."

## **2\. Fix test\_deployment.py Logic**

**Goal:** Ensure the testing script is using the correct execution-plane proxy.

"In deployment/test\_deployment.py, ensure the following:

1. Use agent \= reasoning\_engines.ReasoningEngine(FLAGS.resource\_id) to retrieve the executable proxy.  
2. Replace any reference to stream\_query with stream.  
3. Update the stream input to the standard ADK format: input={'message': user\_input, 'user\_id': FLAGS.user\_id, 'session\_id': session.id}.  
4. Ensure the response parsing correctly iterates through event.get('content', {}).get('parts', \[\]) to find and print the text attribute."

## **3\. Verify Deployment Environment Variables**

**Goal:** Ensure deploy.py passes all necessary context to the remote runtime.

"In deployment/deploy.py, ensure the env\_vars dictionary includes DATASET\_CONFIG\_FILE. Also, add a check to verify that the AGENT\_WHL\_FILE exists and is the correct version (0.1.0) before calling agent\_engines.create. If the wheel is missing, provide a clear error message instructing the user to run uv build \--wheel."

## **4\. Full Cleanup and Redeploy**

**Goal:** A 'nuclear' option to ensure no cached or half-broken agents are causing issues.

"Update deployment/deploy.py logic so that if the \--create flag is used and an agent with the same display\_name already exists, it prompts to delete the old one first. This prevents 'orphaned' reasoning engines in the GCP project. Ensure the AdkApp is initialized with enable\_tracing=True if the WANDB environment variables are present."