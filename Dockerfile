FROM pytorch/pytorch
EXPOSE 8080
ENV DEBIAN_FRONTEND=noninteractive 
COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN apt-get update && apt-get -y install libglib2.0; apt-get clean

COPY . /app
WORKDIR /app
ENTRYPOINT [ "python", "api.py" ]