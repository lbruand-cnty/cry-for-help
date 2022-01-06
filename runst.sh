cd w/streamlit_app/lib
pipenv shell
streamlit run ./srcs/streamlit_app/app.py --server.enableCORS false --server.enableXsrfProtection=false
