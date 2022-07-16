#!/bin/bash
exec gunicorn -b 0.0.0.0:5000 app:app &
exec mlflow server --host 0.0.0.0 --port 5005