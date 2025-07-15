import pandas as pd
import ast
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv('scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            if agg_df.loc[key].diagnostic_class == "NORM" or agg_df.loc[key].diagnostic_class == "MI" or agg_df.loc[key].diagnostic_class == "CD":
                tmp.append(agg_df.loc[key].diagnostic_class)
    
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
columns = ['age', 'sex', 'filename_lr', 'diagnostic_superclass']
Y_FILTERED = Y[Y['diagnostic_superclass'].apply(lambda x: 'NORM' in x or 'MI' in x or 'CD' in x)]
Y_FILTERED = Y_FILTERED[columns]

Y_FILTERED.to_csv("filtered_database.csv")