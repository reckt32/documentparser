#!/bin/bash
# Startup script for Azure App Service

# Start Gunicorn with Flask app
gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 2 app:app
