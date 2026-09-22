import streamlit as st
import pandas as pd
import numpy as np

from model_utils import SUBURBS, PROPERTY_TYPES, train_model

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏡")


@st.cache_resource
def get_model():
    return train_model()


pipe = get_model()

st.title("Sydney Housing Price Predictor")
st.write("Enter a property's details below to get a predicted sale price.")

col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox("Suburb", SUBURBS)
    property_type = st.selectbox("Property type", PROPERTY_TYPES)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
    car_spaces = st.number_input("Car spaces", min_value=0, max_value=10, value=1)

with col2:
    sale_year = st.number_input("Sale year", min_value=2020, max_value=2030, value=2026)

    dont_know_size = st.checkbox("I don't know the size")
    land_size_sqm = None
    floor_area_sqm = None
    if not dont_know_size:
        size_type = st.radio("What size do you have?", ["Land size", "Floor area"])
        if size_type == "Land size":
            land_size_sqm = st.number_input("Land size (sqm)", min_value=0.0, value=400.0)
        else:
            floor_area_sqm = st.number_input("Floor area (sqm)", min_value=0.0, value=100.0)

    atypical_sale = st.checkbox("Unusual sale (development potential, dual-occupancy, etc.)")
    tenanted_sale = st.checkbox("Sold with a tenant in place")

if st.button("Predict price"):
    size_known = int(land_size_sqm is not None or floor_area_sqm is not None)

    row = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'car_spaces': car_spaces,
        'sale_year': sale_year,
        'land_size_sqm': land_size_sqm,
        'floor_area_sqm': floor_area_sqm,
        'atypical_sale': int(atypical_sale),
        'tenanted_sale': int(tenanted_sale),
        'size_known': size_known,
        'suburb': suburb,
        'property_type': property_type,
    }])

    log_pred = pipe.predict(row)
    price = np.expm1(log_pred)[0]

    st.success(f"Estimated sale price: ${price:,.0f}")
    st.caption("This is a model estimate, not a valuation - see Part 4 of the notebook for its known limitations.")
