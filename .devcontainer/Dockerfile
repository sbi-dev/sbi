FROM jupyter/base-notebook:python-3.11
LABEL org.opencontainers.image.source="https://github.com/sbi-dev/sbi"
LABEL org.opencontainers.image.description="Image for a Docker Devcontainer for the sbi toolbox."
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Install latest release of sbi instead of current main branch this is
# intenationally as the codespace shall facilitate as an easy access for new
# users to experience the toolkit rather than a development environment.
USER root

RUN apt update && apt install -y git

RUN pip install \
    sbi \
    ipywidgets
