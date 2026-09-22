import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SUBURBS = ['Mosman', 'Parramatta', 'Campbelltown']
PROPERTY_TYPES = ['House', 'Townhouse', 'Unit']

num_features = ['bedrooms', 'bathrooms', 'car_spaces', 'sale_year']
size_features = ['land_size_sqm', 'floor_area_sqm']
bin_features = ['atypical_sale', 'tenanted_sale', 'size_known']
cat_features = ['suburb', 'property_type']
feature_cols = num_features + size_features + bin_features + cat_features


# same SizeImputer from the notebook (Part 3) - fills land_size_sqm and
# floor_area_sqm per property type instead of one blended median
class SizeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.land_median_by_type_ = X.groupby('property_type')['land_size_sqm'].median()
        self.floor_median_by_type_ = X.groupby('property_type')['floor_area_sqm'].median()
        self.floor_overall_median_ = X['floor_area_sqm'].median()
        return self

    def transform(self, X):
        X = X.copy()
        land_fill = np.where(X['property_type'] == 'Unit', 0,
                              X['property_type'].map(self.land_median_by_type_))
        X['land_size_sqm'] = X['land_size_sqm'].fillna(pd.Series(land_fill, index=X.index))

        floor_group_fill = X['property_type'].map(self.floor_median_by_type_)
        floor_fill = np.where(X['property_type'] == 'House', self.floor_overall_median_, floor_group_fill)
        X['floor_area_sqm'] = X['floor_area_sqm'].fillna(pd.Series(floor_fill, index=X.index))
        return X[['land_size_sqm', 'floor_area_sqm']]

    def get_feature_names_out(self, input_features=None):
        return np.array(['land_size_sqm', 'floor_area_sqm'])


def train_model():
    # trains on the full 105 rows - the train/test split was only needed for
    # Part 3's accuracy check, the deployed app can use all the data we have
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    clean_data = pd.read_csv('cleaned_property_data.csv')
    clean_data['sale_date'] = pd.to_datetime(clean_data['sale_date'])
    clean_data['sale_year'] = clean_data['sale_date'].dt.year
    clean_data['size_known'] = (clean_data['land_size_sqm'].notnull() | clean_data['floor_area_sqm'].notnull()).astype(int)
    clean_data['atypical_sale'] = clean_data['atypical_sale'].astype(int)
    clean_data['tenanted_sale'] = clean_data['tenanted_sale'].astype(int)

    features = clean_data[feature_cols]
    log_target = np.log1p(clean_data['sale_price'])

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('size', Pipeline([('impute', SizeImputer()), ('scale', StandardScaler())]), size_features + ['property_type']),
        ('bin', 'passthrough', bin_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    forest_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    pipe = Pipeline([('prep', preprocess), ('model', forest_model)])
    pipe.fit(features, log_target)
    return pipe
