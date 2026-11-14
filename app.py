"""WSGI entrypoint: `gunicorn app:app` (see startup.sh)."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

from flask import Flask
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from backend.routes import BLUEPRINTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)
CORS(app)
# Respect reverse proxy headers (scheme/host) for correct external URLs
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config['PREFERRED_URL_SCHEME'] = 'https'

for blueprint in BLUEPRINTS:
    app.register_blueprint(blueprint)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
