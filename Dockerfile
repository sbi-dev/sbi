# FROM graphcore/pytorch-jupyter:latest
FROM deepnote/python:3.9-conda
RUN pip install --upgrade pip 
RUN pip install jupyterlab
RUN pip install sbi
WORKDIR /sbi 

# CMD ["python", "-c", "import sbi; print(sbi.__version__)"]

