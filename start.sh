#!/bin/sh
set -e

# Extract images if not already extracted
python extract.py


# Start the Streamlit application
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableXsrfProtection=false