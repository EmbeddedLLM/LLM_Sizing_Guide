# Installation

python setup.py develop

# Deploy

waitress-serve --port 8000 --call 'llmcalculator:create_app'
