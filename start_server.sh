#!/bin/bash
# Run the pipeline script
echo "Running pipeline script..."
python /app/run_pipeline.py

# Now start the ZenML server
echo "Starting ZenML server..."
exec uvicorn zenml.zen_server.zen_server_api:app --host 0.0.0.0 --port 8080
