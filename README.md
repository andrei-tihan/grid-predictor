# Alberta Grid Balance Predictor Capstone Project
By Andrei Tihan, September 2025

This self-directed capstone project for the University of Calgary Machine Learning and AI Certificate focuses on predicting the grid balance in Alberta, Canada, using historical data and machine learning techniques. The goal is to forecast the difference between electricity supply and demand, which is crucial for maintaining grid stability and optimizing energy resources. It aims to give the user a simple breakdown of the predicted balance at a given hour on a given day, as well as a short recommendation of consumption behaviour for consumers and utilities (ex. charge EVs or do laundry at this time, activate Demand Response contingenies). In this project, I used various data sources, including electricity consumption in the form of Alberta Internal Load (AIL) data from the AESO, and generation data from renewable and non-renewable sources, also from the AESO.

While this is a prototype, it provides an example of how the strategic application of AI can solve our most pressing energy challenges. As renewables continue to increase their share of electricity generation, our grids must become more flexible, intelligent, and responsive to maintain grid stability. The Grid Balance Predictor shows how AI can use past data to inform future behaviour and proactively reduce grid stability risks. It can also reduce the costs of the energy transition by optimizing our existing grid and reduicng the need for physical infrastructure buildout.

## Future development
A simple AI chatbot the user can interact with to provide personalized recommendations
Integration of real-time weather data
Interconnection import and export data
Grid stability data such as frequency and voltage

Thanks for checking out my project! If you have any questions or feedback, feel free to reach out.

## Quickstart
1. Create venv and install deps:
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt


2. Put AESO CSVs in `data/raw/`:
- `alberta_net_generation.csv`
- `alberta_internal_load.csv`

3. Run training pipeline (from repo root):
python -m src.train_pipeline --gen data/raw/alberta_net_generation.csv --load data/raw/alberta_internal_load.csv

This writes `data/processed/alberta_hourly_clean.csv`, `data/processed/predictions.csv`, and `models/*`.

4. Run dashboard:
streamlit run app/app.py

This provides a simple UI dashboard to interact with the model and visualize predictions. Currently, this only supports predicting the grid balance for a given hour on a given day for the test set. It will be expanded to include all days of the year.
