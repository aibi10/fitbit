FROM python:3.7

# to install python package psycopg2 (for postgres)
RUN apt-get update

# add user (change to whatever you want)
# prevents running sudo commands

# set current env
ENV HOME /app
WORKDIR /app
ENV PATH="/app/.local/bin:${PATH}"

# Avoid cache purge by adding requirements first
ADD ./requirements.txt ./requirements.txt

COPY . /app

RUN pip3 install --no-cache-dir -r ./requirements.txt

RUN python -m pytest -v tests/test_script.py
# Add the rest of the files

WORKDIR /app
RUN chmod a+x run.sh

EXPOSE 5000
EXPOSE 5005

#RUN mlflow ui
# start web server
#CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
CMD ["./run.sh"]
