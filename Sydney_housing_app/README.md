# Sydney Housing Price Predictor

Streamlit app that predicts a sale price for a property in Mosman, Parramatta or Campbelltown, using the Random Forest model from the notebook.

## Setup
```
pip install -r requirements.txt
streamlit run app.py
```
That's it - it should open in your browser at localhost:8501.

## Notes

- The app trains the model itself when it starts, instead of
  loading a saved model file. I tried loading a trained model first, but it broke when run on a different computer with a different scikit-learn version.
  Training fresh each time avoids that problem completely.
- `cleaned_property_data.csv` needs to stay in the same folder as `app.py`that's what
  it trains on.
- This gives an estimate, not a valuation. See Part 4 of the notebook for where the model
  tends to be less accurate.
