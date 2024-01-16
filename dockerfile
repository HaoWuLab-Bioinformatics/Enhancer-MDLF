FROM python:3.8.18
WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt
RUN chmod +x /code/start_project.sh
CMD ["/code/start_project.sh"]
