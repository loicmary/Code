from entsoe import EntsoePandasClient
import pandas as pd
import argparse
from functools import reduce


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-start', required=True, type=str, help='Start date for the data')
    parser.add_argument('-end', required=True, type=str, help='End date for the data')
    parser.add_argument('-study_country', required=True, type=str, help='Country code for the country you want to study')
    parser.add_argument('-borders', required=True, nargs='*',type=str, help='Country codes of the borders of the country you want to study')
    parser.add_argument('-tz', required=True, type=str, help='Timezone of the country')
    args = parser.parse_args()
    #print((type(args.start), type(args.tz)))
    print((args.start, args.tz))

    start = pd.Timestamp(args.start, tz=args.tz)
    end = pd.Timestamp(args.end, tz=args.tz)
    study_country = args.study_country
    borders = args.borders

    client = EntsoePandasClient(api_key=api_key)

    imports = []
    exports = []
    print('Start querying....')
    for c in borders :
        im = client.query_crossborder_flows(country_code_from=c, country_code_to=study_country, start=start, end=end).reset_index()
        exp = client.query_crossborder_flows(country_code_from=study_country, country_code_to=c, start=start, end=end).reset_index()

        imports.append(im)
        exports.append(exp)

    
    imports_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='index'), imports)
    exports_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='index'), exports)
    flow = exports_df.loc[:, ~exports_df.columns.isin(['index'])].sum(axis=1)- imports_df.loc[:, ~imports_df.columns.isin(['index'])].sum(axis=1)
    flow_df = pd.DataFrame({'index': exports_df['index'], 'flow': flow})

    flow_df.to_csv('flows.csv', sep=';', index=False)
    print('dataset created')