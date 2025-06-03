import numpy as np
import pandas as pd




def convert_labels(regression_df: pd.DataFrame):
    classification_df = regression_df.__deepcopy__() 
    print(classification_df.head())
    classification_df = classification_df.drop(columns=regression_df.columns[-8:])
    classification_labels = np.argmax(regression_df[regression_df.columns[-8:]].to_numpy(), axis=1)
    classification_df['labels'] = classification_labels
    return classification_df

    


