#!/bin/bash
# Load environment variables from .env
if [ -f .env ]; then
    # Filter out comments and empty lines before exporting
    # Using 'set -a' for simpler env var handling
    set -a
    source <(grep -v '^#' .env | sed 's/#.*//')
    set +a
fi

# Use the image we built previously or build it again if needed
IMAGE_TAG="gcr.io/$GOOGLE_CLOUD_PROJECT/data-science-agent"

# Build the image
gcloud builds submit --tag "$IMAGE_TAG" .

# Deploy to Cloud Run
gcloud run deploy data-science-agent \
    --image "$IMAGE_TAG" \
    --port 8080 \
    --memory 2G \
    --service-account "ds-adk-agent-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --set-cloudsql-instances "$GOOGLE_CLOUD_PROJECT:$GOOGLE_CLOUD_LOCATION:ds-agent-session-service" \
    --set-env-vars "MCP_SERVER_URL=https://mcp-server-396631018769.us-central1.run.app/mcp" \
    --set-env-vars "GOOGLE_API_KEY=$GOOGLE_API_KEY" \
    --set-env-vars "GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION" \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT" \
    --set-env-vars "APP_NAME=$APP_NAME" \
    --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=$GOOGLE_GENAI_USE_VERTEXAI" \
    --set-env-vars "BQ_NL2SQL_METHOD=$BQ_NL2SQL_METHOD" \
    --set-env-vars "BQ_COMPUTE_PROJECT_ID=$BQ_COMPUTE_PROJECT_ID" \
    --set-env-vars "BQ_DATA_PROJECT_ID=$BQ_DATA_PROJECT_ID" \
    --set-env-vars "BQ_DATASET_ID=$BQ_DATASET_ID" \
    --set-env-vars "DATASET_CONFIG_FILE=$DATASET_CONFIG_FILE" \
    --set-env-vars "ALLOYDB_TOOLSET=$ALLOYDB_TOOLSET" \
    --set-env-vars "ALLOYDB_SCHEMA_NAME=$ALLOYDB_SCHEMA_NAME" \
    --set-env-vars "ALLOYDB_DATABASE=$ALLOYDB_DATABASE" \
    --set-env-vars "ALLOYDB_PROJECT_ID=$ALLOYDB_PROJECT_ID" \
    --set-env-vars "ALLOYDB_AGENT_MODEL=$ALLOYDB_AGENT_MODEL" \
    --set-env-vars "MCP_TOOLBOX_HOST=$MCP_TOOLBOX_HOST" \
    --set-env-vars "MCP_TOOLBOX_PORT=$MCP_TOOLBOX_PORT" \
    --set-env-vars "CROSS_DATASET_RELATIONS_DEFS=$CROSS_DATASET_RELATIONS_DEFS" \
    --set-env-vars "ROOT_AGENT_MODEL=$ROOT_AGENT_MODEL" \
    --set-env-vars "ANALYTICS_AGENT_MODEL=$ANALYTICS_AGENT_MODEL" \
    --set-env-vars "BIGQUERY_AGENT_MODEL=$BIGQUERY_AGENT_MODEL" \
    --set-env-vars "BASELINE_NL2SQL_MODEL=$BASELINE_NL2SQL_MODEL" \
    --set-env-vars "CHASE_NL2SQL_MODEL=$CHASE_NL2SQL_MODEL" \
    --set-env-vars "BQML_AGENT_MODEL=$BQML_AGENT_MODEL" \
    --set-env-vars "SERVE_WEB_INTERFACE=True" \
    --set-env-vars "SESSION_SERVICE_URI=postgresql+pg8000://postgres:ds-agent-demo@/postgres?unix_sock=/cloudsql/$GOOGLE_CLOUD_PROJECT:$GOOGLE_CLOUD_LOCATION:ds-agent-session-service/.s.PGSQL.5432" \
    --region "$GOOGLE_CLOUD_LOCATION" \
    --allow-unauthenticated \
    --quiet
