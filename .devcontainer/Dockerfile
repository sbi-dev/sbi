FROM jupyter/base-notebook:python-${VARIANT}

# Install latest release of sbi instead of current main branch this is
# intenationally as the codespace shall facilitate as an easy access for new
# users to experience the toolkit rather than a development environment. 
RUN pip install --upgrade pip && \
    pip install sbi 
