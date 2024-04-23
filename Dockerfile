# Use a specific version of Python as the base image
FROM python:3.9-slim AS base

# Set environment variables to configure Python and pip
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    ZENML_CONTAINER=1

# Upgrade pip and install ZenML with necessary extras
RUN pip install --upgrade pip && \
    pip install zenml[server,secrets-gcp,gcsfs,connectors-gcp]

# Set the working directory
WORKDIR /app

# Copy your project files and the entrypoint script
COPY . /app
COPY start_server.sh /app/start_server.sh

# Install any required dependencies
RUN pip install -r requirements.txt

# Initialize ZenML and configure components
RUN zenml init

# Environment and user setup as before
ENV ZENML_CONFIG_PATH=/app/.zenconfig \
    ZENML_DEBUG=false \
    ZENML_ANALYTICS_OPT_IN=true
ARG USERNAME=zenml
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    chown -R $USER_UID:$USER_GID /app
USER $USERNAME

RUN zenml integration install mlflow -y && \
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow && \
    zenml model-deployer register mlflow --flavor=mlflow && \
    zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set

# Expose the server port
EXPOSE 8080

# Use the entrypoint script
ENTRYPOINT ["/app/start_server.sh"]
