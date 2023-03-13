
import numpy as np, pandas as pd
import sys 

from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns, selector_type='keep'):
        self.columns = columns
        self.selector_type = selector_type
        
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        if self.selector_type == 'keep':
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == 'drop':
            dropped_cols = [col for col in X.columns if col in self.columns]  
            X = X.drop(dropped_cols, axis=1)      
        else: 
            raise Exception(f'''
                Error: Invalid selector_type. 
                Allowed values ['keep', 'drop']
                Given type = {self.selector_type} ''')
        
        return X
    
    
class TypeCaster(BaseEstimator, TransformerMixin):  
    def __init__(self, vars, cast_type):
        super().__init__()
        self.vars = vars
        self.cast_type = cast_type
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):  
        data = data.copy()
        applied_cols = [col for col in self.vars if col in data.columns] 
        for var in applied_cols: 
            data[var] = data[var].apply(self.cast_type)
        return data


class StringTypeCaster(TypeCaster):  
    ''' Casts categorical features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, cat_vars): 
        super(StringTypeCaster, self).__init__(cat_vars, str)


class FloatTypeCaster(TypeCaster):  
    ''' Casts float features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, num_vars):
        super(FloatTypeCaster, self).__init__(num_vars, float)



class OneHotEncoderMultipleCols(BaseEstimator, TransformerMixin):  
    def __init__(self, ohe_columns, max_num_categories=10): 
        super().__init__()
        self.ohe_columns = ohe_columns
        self.max_num_categories = max_num_categories
        self.top_cat_by_ohe_col = {}
        
        
    def fit(self, X, y=None):    
        for col in self.ohe_columns:
            if col in X.columns: 
                self.top_cat_by_ohe_col[col] = [ 
                    cat for cat in X[col].value_counts()\
                        .sort_values(ascending = False).head(self.max_num_categories).index
                    ]         
        return self
    
    
    def transform(self, data): 
        data.reset_index(inplace=True, drop=True)
        df_list = [data]
        cols_list = list(data.columns)
        for col in self.ohe_columns:
            if len(self.top_cat_by_ohe_col[col]) > 0:
                if col in data.columns:                
                    for cat in self.top_cat_by_ohe_col[col]:
                        col_name = col + '_' + cat
                        # data[col_name] = np.where(data[col] == cat, 1, 0)
                        vals = np.where(data[col] == cat, 1, 0)
                        df = pd.DataFrame(vals, columns=[col_name])
                        df_list.append(df)
                        
                        cols_list.append(col_name)
                else: 
                    raise Exception(f'''
                        Error: Fitted one-hot-encoded column {col}
                        does not exist in dataframe given for transformation.
                        This will result in a shape mismatch for train/prediction job. 
                        ''')
        transformed_data = pd.concat(df_list, axis=1, ignore_index=True) 
        transformed_data.columns =  cols_list
        return transformed_data


class ValueClipper(BaseEstimator, TransformerMixin): 
    def __init__(self, fields_to_clip, min_val, max_val) -> None:
        super().__init__()
        self.fields_to_clip = fields_to_clip
        self.min_val = min_val
        self.max_val = max_val
    
    def fit(self, data): return self
    
    def transform(self, data): 
        for field in self.fields_to_clip:
            if self.min_val is not None: 
                data[field] = data[field].clip(lower=self.min_val)
            if self.max_val is not None: 
                data[field] = data[field].clip(upper=self.max_val)
        return data