# LLM Calculator

This is a calculator that is designed with the following questions in mind:

1. What is the minimum number of GPUs (of a type of GPU) that I need to rent/buy to serve model X?
2. What is the performance that I could get from using N number of GPUs when serving model X?
3. What is the minimum number of GPUs (of a type of GPU) that I need to rent/buy to serve model X with targeted prefill speed and time-per-output-token (TPOT)?
4. How long does it take to serve a long context length model?

# Installation

python setup.py develop

# Debug 

flask --app llmcalculator run --debug

# Deploy

waitress-serve --port 8000 --call 'llmcalculator:create_app'


