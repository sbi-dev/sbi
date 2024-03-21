FROM deepnote/python:3.9-conda
RUN pip install --upgrade pip 
RUN pip install jupyterlab
# install from pip
# RUN pip install --upgrade sbi
# install from source
RUN pip install git+https://github.com/sbi-dev/sbi.git
WORKDIR /sbi 

# CMD ["python", "-c", "import sbi; print(sbi.__version__)"]

