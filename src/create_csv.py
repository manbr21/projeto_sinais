import pandas as pd
import ast

src_path = "../database_csv/"
dest_path = "../generated_csv/"

sampling_rate=100

Y = pd.read_csv(src_path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

agg_df = pd.read_csv(src_path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    app = []

    for value in y_dic.values():
        if value != 0:
            app.append(value)
    
    if len(app) == 1:
        for key in y_dic.keys():
            if key in agg_df.index:
                if (agg_df.loc[key].diagnostic_class == "NORM" or agg_df.loc[key].diagnostic_class == "MI" or agg_df.loc[key].diagnostic_class == "CD"):
                    tmp.append(agg_df.loc[key].diagnostic_class)
    
        return list(set(tmp))

    return []

Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
columns = ['age', 'sex', 'filename_lr', 'diagnostic_superclass']
Y_FILTERED = Y[Y['diagnostic_superclass'].apply(lambda x: 'NORM' in x or 'MI' in x or 'CD' in x)]
Y_FILTERED = Y_FILTERED[columns]

Y_FILTERED.to_csv(dest_path + "filtered_database.csv")