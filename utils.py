import pandas as pd
from tqdm import tqdm_notebook

json_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
TOP_N = 3


def cast_crew_features(movies_df):
    for team in tqdm_notebook(['crew', 'cast']):
        team_df = movies_df[['release_date', 'revenue', 'budget', team]].explode(team)
        team_df = team_df.dropna(subset=[team])
        _team_metedata = pd.DataFrame(list(team_df[team])).rename(columns={'id': 'member_id'})
        team_df = pd.concat([team_df.reset_index(), _team_metedata], axis=1)
        if team == 'cast':
            team_df = team_df[team_df.order < TOP_N]
        elif team == 'crew':
            team_df = team_df[team_df.job == 'Director']

        for feature_name in tqdm_notebook(['profit', 'revenue', 'power_score']):
            feature_name_ext = feature_name + ('_actor' if team == 'cast' else '_director')
            team_df[feature_name_ext] = None
            for idx, row in tqdm_notebook(team_df.iterrows(), total=len(team_df)):
                relevant_movies = team_df[
                    (team_df.release_date <= row.release_date) & (team_df.member_id == row.member_id)
                    ]
                if feature_name == 'power_score':
                    team_df.loc[idx, feature_name_ext] = len(relevant_movies)
                if feature_name == 'revenue':
                    team_df.loc[idx, feature_name_ext] = relevant_movies.revenue.mean()
                if feature_name == 'profit':
                    team_df.loc[idx, feature_name_ext] = (relevant_movies.revenue - relevant_movies.budget).mean()

            new_features_df = team_df.groupby('movie_id')[[feature_name_ext]].agg(
                {feature_name_ext: ['mean', 'sum', 'max', 'min']})
            if team == 'cast':
                new_features_df.columns = [f'_top_{TOP_N}_'.join(col) for col in new_features_df.columns]
            elif team == 'crew':
                new_features_df.columns = [f'_'.join(col) for col in new_features_df.columns]
            movies_df = movies_df.join(new_features_df)
    return movies_df


def load_data():
    movies_df = pd.read_csv('data/movies.csv')
    movies_df.index = movies_df['id']
    movies_df.index = movies_df.index.rename(name='movie_id')

    cci_df = pd.read_csv('data/macroeconomics/Consumer_Confidence_Index_2.csv')[['iso_3166_1_alpha_3', 'TIME', 'CCI']]
    cpi_df = pd.read_csv('data/macroeconomics/Consumer_Price_Index.csv').drop(
        columns=['Indicator Name', 'Indicator Code'])
    gdp_df = pd.read_csv('data/macroeconomics/GDP_per_capita.csv').drop(columns=['Indicator Name', 'Indicator Code'])
    rir_df = pd.read_csv('data/macroeconomics/Real_Interest_Rate_2.csv').drop(columns=['Indicator Name', 'Indicator Code'])
    uer_df = pd.read_csv('data/macroeconomics/Unemployment_Rate_2.csv').drop(columns=['Indicator Name', 'Indicator Code'])

    eacc_df = pd.read_csv('data/macroeconomics/EURO_AREA_COUNTR_CODES.csv')
    rotwcc_df = pd.read_csv('data/macroeconomics/REST_OF_THE_WORLD_COUNTRY_CODES.csv')

    columns_to_drop = [
        'homepage',
        'imdb_id',
        'original_title',
        'overview',
        'popularity',
        'poster_path',
        #     'spoken_languages',
        'status',
        'tagline',
        'title',
    ]

    movies_df = movies_df.drop(columns=columns_to_drop)

    return movies_df, cci_df, cpi_df, gdp_df, rir_df, uer_df, eacc_df, rotwcc_df


def date_features(movies_df):
    movies_df.belongs_to_collection = ~movies_df.belongs_to_collection.isna()
    movies_df.release_date = pd.to_datetime(movies_df.release_date)

    movies_df['year'] = movies_df.release_date.dt.year.astype(int)
    movies_df['month'] = movies_df.release_date.dt.month.astype(int)
    movies_df['day'] = movies_df.release_date.dt.day.astype(int)
    movies_df['dayofweek'] = movies_df.release_date.dt.dayofweek.astype(int)
    movies_df['quarter'] = movies_df.release_date.dt.quarter.astype(int)

    movies_df.year = movies_df.year.apply(lambda x: x - 100 if x >= 2021 else x)
    movies_df.release_date = pd.to_datetime(movies_df[['year', 'month', 'day']])
    # movies_df = movies_df[movies_df.year >= 1961]

    return movies_df


def eval_dict_columns(movies_df):
    def get_dictionary(s):
        try:
            d = eval(s)
        except:
            d = {}
        return d

    for col in json_cols:
        movies_df[col] = movies_df[col].apply(lambda x: get_dictionary(x))
    return movies_df


def get_json_dict(movies_df):
    result = dict()
    for e_col in json_cols:
        d = dict()
        rows = movies_df[e_col].values
        for row in rows:
            if row is None: continue
            for i in row:
                if i['name'] not in d:
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d
    return result


def categorical_cut(movies_df, cutoff_thresholds):
    movies_dict = get_json_dict(movies_df)
    results_summary = []

    for col in json_cols:
        len_before = len(movies_dict[col])

        for val, cnt in list(movies_dict[col].items()):
            if cnt < cutoff_thresholds[col]:
                del movies_dict[col][val]

        results_summary.append({
            'name': col,
            'before': len_before,
            'after': len(movies_dict[col])
        })

    results_summary_df = pd.DataFrame(results_summary).set_index('name')
    movies_df['iso_3166_1_alpha_2'] = movies_df['production_countries'].apply(
        lambda x: ",".join(r['iso_3166_1'] for r in x))

    for col in ['genres', 'production_countries', 'spoken_languages', 'production_companies', 'Keywords']:
        movies_df[col] = movies_df[col].map(
            lambda x: sorted(
                list(set([n if n in movies_dict[col] else col + '_etc' for n in [d['name'] for d in x]])))).map(
            lambda x: ','.join(map(str, x)))
        temp = movies_df[col].str.get_dummies(sep=',')
        movies_df = pd.concat([movies_df, temp], axis=1, sort=False)
    return movies_df, results_summary_df


def countr_code2country_category(series, encoding, eacc_df, rotwcc_df):
    # encoding 2 or 3
    if encoding == 2:
        encoding_column = 'iso_3166_1_alpha_2'
    elif encoding == 3:
        encoding_column = 'iso_3166_1_alpha_3'
    else:
        assert False, "ENCODING MUST BE EITHER 2 or 3 (int)"

    new_series = series.copy()
    for i, row in series.items():
        if 'US' in row:
            new_series[i] = 'USA'
        elif any(eacc in row for eacc in eacc_df[encoding_column]):
            new_series[i] = 'EUR'
        elif any(rotwcc in row for rotwcc in rotwcc_df[encoding_column]):
            new_series[i] = 'ROW'
        else:
            new_series[i] = 'OTHER'

    return new_series


def join_macroeconomics(movies_df, cci_df, cpi_df, gdp_df, rir_df, uer_df, eacc_df, rotwcc_df):
    movies_df['country_category'] = countr_code2country_category(movies_df.iso_3166_1_alpha_2, 2, eacc_df, rotwcc_df)
    movies_df['country_category'] = movies_df['country_category'].replace({"OTHER": "ROW"})

    cci_df['country_category'] = countr_code2country_category(cci_df.iso_3166_1_alpha_3, 3, eacc_df, rotwcc_df)
    cci_df['year'] = cci_df.TIME.apply(lambda x: x.split("-")[0])
    cci_df.year = cci_df.year.astype(int)
    cci_df = cci_df.groupby(['country_category', 'year']).agg(
        cci=('CCI', 'mean')
    )

    cpi_df['country_category'] = countr_code2country_category(cpi_df.iso_3166_1_alpha_3, 3, eacc_df, rotwcc_df)
    cpi_df = pd.melt(cpi_df.iloc[:, 2:], id_vars=['country_category'], var_name='year', value_name='cpi')
    cpi_df = cpi_df.dropna()
    cpi_df.year = cpi_df.year.astype(int)
    cpi_df = cpi_df.groupby(['country_category', 'year']).agg(
        cpi=('cpi', 'mean')
    )

    gdp_df['country_category'] = countr_code2country_category(gdp_df.iso_3166_1_alpha_3, 3, eacc_df, rotwcc_df)
    gdp_df = pd.melt(gdp_df.iloc[:, 2:], id_vars=['country_category'], var_name='year', value_name='gdp')
    gdp_df = gdp_df.dropna()
    gdp_df.year = gdp_df.year.astype(int)
    gdp_df = gdp_df.groupby(['country_category', 'year']).agg(
        gdp=('gdp', 'mean')
    )

    rir_df['country_category'] = countr_code2country_category(rir_df.iso_3166_1_alpha_3, 3, eacc_df, rotwcc_df)
    rir_df = pd.melt(rir_df.iloc[:, 2:], id_vars=['country_category'], var_name='year', value_name='rir')
    rir_df = rir_df.dropna()
    rir_df.year = rir_df.year.astype(int)
    rir_df = rir_df.groupby(['country_category', 'year']).agg(
        rir=('rir', 'mean')
    )

    uer_df['country_category'] = countr_code2country_category(uer_df.iso_3166_1_alpha_3, 3, eacc_df, rotwcc_df)
    uer_df = pd.melt(uer_df.iloc[:, 2:], id_vars=['country_category'], var_name='year', value_name='uer')
    uer_df = uer_df.dropna()
    uer_df.year = uer_df.year.astype(int)
    uer_df = uer_df.groupby(['country_category', 'year']).agg(
        uer=('uer', 'mean')
    )
    movies_df = movies_df.reset_index().merge(cci_df, how='left', on=['year', 'country_category']).set_index('movie_id')
    movies_df = movies_df.reset_index().merge(cpi_df, how='left', on=['year', 'country_category']).set_index('movie_id')
    movies_df = movies_df.reset_index().merge(gdp_df, how='left', on=['year', 'country_category']).set_index('movie_id')
    movies_df = movies_df.reset_index().merge(rir_df, how='left', on=['year', 'country_category']).set_index('movie_id')
    movies_df = movies_df.reset_index().merge(uer_df, how='left', on=['year', 'country_category']).set_index('movie_id')
    movies_indexes = movies_df.index

    return movies_df


def prepare_additional_features(movies_df):
    # original features from source notebooks

    movies_df['genders_0_crew'] = movies_df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    movies_df['genders_1_crew'] = movies_df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    movies_df['genders_2_crew'] = movies_df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    movies_df['num_Keywords'] = movies_df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    movies_df['num_cast'] = movies_df['cast'].apply(lambda x: len(x) if x != {} else 0)
    movies_df['num_crew'] = movies_df['crew'].apply(lambda x: len(x))

    # nasze ok
    movies_df['runtime_categories'] = movies_df.runtime.apply(lambda x: "short" if x < 81 else ("medium" if x < 131 else "long"))
    movies_df = pd.concat([movies_df, pd.get_dummies(movies_df.pop('runtime_categories'), prefix='runtime_')], axis=1)
    movies_df['_budget_runtime_ratio'] = movies_df['budget'] / movies_df['runtime']
    mean_runtime_by_year = movies_df.groupby("year")[["runtime"]].aggregate(mean_ryntime_by_year=('runtime', 'mean'))
    movies_df = movies_df.join(mean_runtime_by_year, 'year', 'left')
    mean_budget_by_year = movies_df.groupby("year")[["budget"]].aggregate(mean_budget_by_year=('budget', 'mean'))
    movies_df = movies_df.join(mean_budget_by_year, 'year', 'left')

    movies_df['isOriginalLanguageEng'] = 0
    movies_df.loc[movies_df['original_language'] == "en", "isOriginalLanguageEng"] = 1
    movies_df['production_countries_count'] = movies_df['production_countries'].apply(lambda x: len(x))
    movies_df['production_companies_count'] = movies_df['production_companies'].apply(lambda x: len(x))

    # movies_df.fillna(value=0.0, inplace=True)

    return movies_df


def drop_columns(movies_df):
    movies_df = movies_df.drop([
        'genres', 'homepage', 'imdb_id', 'overview'
        , 'poster_path', 'production_companies', 'release_date', 'spoken_languages'
        , 'status', 'title', 'Keywords', 'cast', 'crew', 'original_language',
        'original_title', 'tagline', 'collection_id', 'movie_id'
    ], axis=1, errors='ignore')
    return movies_df


def clean_data(movies_df):
    # my changes to original cleaning above

    movies_df.loc[16, 'revenue'] = 192864
    movies_df.loc[90, 'budget'] = 30000000
    movies_df.loc[118, 'budget'] = 60000000
    movies_df.loc[149, 'budget'] = 18000000
    movies_df.loc[313, 'revenue'] = 12000000
    movies_df.loc[451, 'revenue'] = 12000000
    movies_df.loc[464, 'budget'] = 20000000
    movies_df.loc[470, 'budget'] = 13000000
    movies_df.loc[513, 'budget'] = 930000
    movies_df.loc[797, 'budget'] = 8000000
    movies_df.loc[819, 'budget'] = 90000000
    movies_df.loc[850, 'budget'] = 90000000
    movies_df.loc[1007, 'budget'] = 2000000
    movies_df.loc[1112, 'budget'] = 7500000
    movies_df.loc[1131, 'budget'] = 4300000
    movies_df.loc[1359, 'budget'] = 10000000
    movies_df.loc[1542, 'budget'] = 1000000
    movies_df.loc[1542, 'revenue'] = 3514780
    movies_df.loc[1570, 'budget'] = 15800000
    movies_df.loc[1571, 'budget'] = 4000000
    movies_df.loc[1714, 'budget'] = 46000000
    movies_df.loc[1721, 'budget'] = 17500000
    movies_df.loc[1865, 'revenue'] = 25000000
    movies_df.loc[1885, 'budget'] = 12000000
    movies_df.loc[1885, 'revenue'] = 23693646
    movies_df.loc[2091, 'budget'] = 10000000
    movies_df.loc[2268, 'budget'] = 17500000
    movies_df.loc[2491, 'budget'] = 6000000
    movies_df.loc[2491, 'revenue'] = 6849998
    movies_df.loc[2602, 'budget'] = 31000000
    movies_df.loc[2612, 'budget'] = 15000000
    movies_df.loc[2696, 'budget'] = 10000000
    movies_df.loc[2801, 'budget'] = 10000000
    movies_df.loc[335, 'budget'] = 649730
    movies_df.loc[335, 'revenue'] = 3898383
    movies_df.loc[348, 'budget'] = 12031012
    movies_df.loc[348, 'revenue'] = 1607836
    movies_df.loc[470, 'budget'] = 13000000
    movies_df.loc[513, 'budget'] = 1100000
    movies_df.loc[640, 'budget'] = 6000000
    movies_df.loc[640, 'revenue'] = 10558970
    movies_df.loc[696, 'budget'] = 1299461
    movies_df.loc[696, 'revenue'] = 3189267
    movies_df.loc[797, 'budget'] = 8000000
    movies_df.loc[850, 'budget'] = 1500000
    movies_df.loc[1199, 'budget'] = 5000000
    movies_df.loc[1199, 'revenue'] = 104268727
    movies_df.loc[1282, 'budget'] = 9000000
    movies_df.loc[1282, 'revenue'] = 48977233
    movies_df.loc[1347, 'budget'] = 1000000
    movies_df.loc[1347, 'revenue'] = 5000000
    movies_df.loc[1755, 'budget'] = 2000000
    movies_df.loc[1755, 'revenue'] = 1125910
    movies_df.loc[1801, 'budget'] = 5000000
    movies_df.loc[1801, 'revenue'] = 135280
    movies_df.loc[1918, 'budget'] = 1605000
    movies_df.loc[1918, 'revenue'] = 4500000
    movies_df.loc[2033, 'budget'] = 4000000
    movies_df.loc[2033, 'revenue'] = 20000000
    movies_df.loc[2118, 'budget'] = 344000
    movies_df.loc[2118, 'revenue'] = 400000
    movies_df.loc[2252, 'budget'] = 7800000
    movies_df.loc[2252, 'revenue'] = 49285374
    movies_df.loc[2256, 'budget'] = 14000000
    movies_df.loc[2256, 'revenue'] = 6500000
    movies_df.loc[2696, 'budget'] = 10000000
    movies_df.loc[2696, 'revenue'] = 400000

    ## budget values substituted by me 

    # 22.05.07
    movies_df.loc[8, 'budget'] = 60000
    movies_df.loc[9, 'budget'] = 31000000
    movies_df.loc[12, 'budget'] = 10000000
    movies_df.loc[18, 'budget'] = 12000000
    movies_df.loc[23, 'budget'] = 242000
    movies_df.loc[32, 'budget'] = 8000000
    movies_df.loc[32, 'revenue'] = 16200000
    movies_df.loc[34, 'budget'] = 130000
    movies_df.loc[34, 'revenue'] = 106980
    movies_df.loc[39, 'budget'] = 70000
    movies_df.loc[49, 'budget'] = 5000000
    movies_df.loc[53, 'budget'] = 7000000
    movies_df.loc[54, 'budget'] = 10000000
    movies_df.loc[56, 'budget'] = 8000000
    movies_df.loc[63, 'budget'] = 40000000
    movies_df.loc[74, 'budget'] = 800839
    movies_df.loc[74, 'revenue'] = 54946
    movies_df.loc[79, 'budget'] = 17500000
    movies_df.loc[92, 'revenue'] = 6000000
    movies_df.loc[94, 'budget'] = 10000000
    movies_df.loc[98, 'budget'] = 3900000
    movies_df.loc[104, 'budget'] = 19000000
    movies_df.loc[105, 'budget'] = 10900000
    movies_df.loc[117, 'budget'] = 1600000
    movies_df.loc[117, 'revenue'] = 2800000
    movies_df.loc[119, 'budget'] = 2000000
    movies_df.loc[127, 'budget'] = 9000000
    movies_df.loc[142, 'budget'] = 12000000
    movies_df.loc[147, 'budget'] = 14000000
    movies_df.loc[171, 'budget'] = 7500000
    movies_df.loc[172, 'budget'] = 5500000
    movies_df.loc[190, 'budget'] = 8000000
    movies_df.loc[194, 'budget'] = 700000
    movies_df.loc[207, 'budget'] = 2500000
    movies_df.loc[213, 'budget'] = 15000000
    movies_df.loc[222, 'budget'] = 750000
    movies_df.loc[226, 'budget'] = 30000000
    movies_df.loc[236, 'budget'] = 35500000
    movies_df.loc[244, 'budget'] = 20000000
    movies_df.loc[245, 'budget'] = 3000000
    movies_df.loc[259, 'budget'] = 110000000
    movies_df.loc[261, 'budget'] = 3000000
    movies_df.loc[270, 'budget'] = 32000
    movies_df.loc[281, 'budget'] = 5250000
    movies_df.loc[288, 'budget'] = 30000000
    movies_df.loc[303, 'budget'] = 11000000
    movies_df.loc[304, 'budget'] = 12000000
    movies_df.loc[305, 'budget'] = 9000000
    movies_df.loc[332, 'budget'] = 2711391
    movies_df.loc[341, 'budget'] = 4317424
    movies_df.loc[353, 'budget'] = 6000000
    movies_df.loc[361, 'budget'] = 80000
    movies_df.loc[361, 'revenue'] = 5631241
    movies_df.loc[368, 'budget'] = 42000000
    movies_df.loc[371, 'budget'] = 1579456
    movies_df.loc[380, 'budget'] = 6500000
    movies_df.loc[384, 'budget'] = 40600000
    movies_df.loc[397, 'budget'] = 2500000
    movies_df.loc[398, 'budget'] = 5000000
    movies_df.loc[404, 'budget'] = 18000000
    movies_df.loc[405, 'budget'] = 8500000
    movies_df.loc[426, 'budget'] = 4000000
    movies_df.loc[428, 'budget'] = 2637075
    movies_df.loc[442, 'budget'] = 9600000
    movies_df.loc[444, 'budget'] = 10000000
    movies_df.loc[446, 'budget'] = 4000000
    movies_df.loc[452, 'budget'] = 6000000
    movies_df.loc[455, 'budget'] = 9000000
    movies_df.loc[466, 'budget'] = 6000000
    movies_df.loc[476, 'budget'] = 8000000
    movies_df.loc[480, 'budget'] = 25000000
    movies_df.loc[485, 'budget'] = 10000000
    movies_df.loc[494, 'budget'] = 10000000
    movies_df.loc[499, 'budget'] = 6500000
    movies_df.loc[499, 'revenue'] = 25500000
    movies_df.loc[281, 'revenue'] = 10200000
    movies_df.loc[153, 'revenue'] = 240000
    movies_df.loc[151, 'revenue'] = 15000000
    movies_df.loc[132, 'revenue'] = 17000

    # 22.05.08
    movies_df.loc[505, 'budget'] = 2000000
    movies_df.loc[540, 'budget'] = 1500000
    movies_df.loc[554, 'budget'] = 19000000
    movies_df.loc[556, 'budget'] = 25000000
    movies_df.loc[557, 'budget'] = 12000000
    movies_df.loc[562, 'budget'] = 1000000
    movies_df.loc[566, 'budget'] = 925462
    movies_df.loc[576, 'budget'] = 15547770
    movies_df.loc[580, 'budget'] = 1500000
    movies_df.loc[581, 'budget'] = 9000000
    movies_df.loc[582, 'budget'] = 16000000
    movies_df.loc[585, 'budget'] = 5000000
    movies_df.loc[591, 'budget'] = 3000000
    movies_df.loc[595, 'budget'] = 20000000
    movies_df.loc[620, 'budget'] = 6000000
    movies_df.loc[627, 'budget'] = 12000000
    movies_df.loc[630, 'budget'] = 7500000
    movies_df.loc[635, 'budget'] = 1200000
    movies_df.loc[637, 'budget'] = 652675
    movies_df.loc[653, 'budget'] = 20000000
    movies_df.loc[658, 'budget'] = 8000000
    movies_df.loc[664, 'budget'] = 1000000
    movies_df.loc[665, 'budget'] = 15000000
    movies_df.loc[668, 'budget'] = 5000000
    movies_df.loc[670, 'budget'] = 30000000
    movies_df.loc[679, 'budget'] = 20600000
    movies_df.loc[683, 'budget'] = 20000000
    movies_df.loc[695, 'budget'] = 10000000
    movies_df.loc[697, 'budget'] = 40000000
    movies_df.loc[711, 'budget'] = 11000000
    movies_df.loc[717, 'budget'] = 6000000
    movies_df.loc[719, 'budget'] = 24000000
    movies_df.loc[720, 'budget'] = 4000000
    movies_df.loc[723, 'budget'] = 17000000

    movies_df.loc[733, 'budget'] = 2000000
    movies_df.loc[749, 'budget'] = 25000000
    movies_df.loc[759, 'budget'] = 8400000
    movies_df.loc[765, 'budget'] = 15000000
    movies_df.loc[775, 'budget'] = 2000000
    movies_df.loc[780, 'budget'] = 24000000
    movies_df.loc[785, 'budget'] = 25000000
    movies_df.loc[809, 'budget'] = 370185
    movies_df.loc[812, 'budget'] = 8500000
    movies_df.loc[824, 'budget'] = 1250000
    movies_df.loc[836, 'budget'] = 14500000
    movies_df.loc[837, 'budget'] = 12000000
    movies_df.loc[839, 'budget'] = 14000000
    movies_df.loc[842, 'budget'] = 15000000
    movies_df.loc[854, 'budget'] = 13000000
    movies_df.loc[868, 'budget'] = 1480740
    movies_df.loc[877, 'budget'] = 15000000
    movies_df.loc[879, 'budget'] = 18000000
    movies_df.loc[886, 'budget'] = 12000000
    movies_df.loc[887, 'budget'] = 38171
    movies_df.loc[887, 'revenue'] = 97336
    movies_df.loc[891, 'budget'] = 14750000
    movies_df.loc[899, 'budget'] = 35000000
    movies_df.loc[910, 'budget'] = 2000000
    movies_df.loc[918, 'budget'] = 10000000
    movies_df.loc[921, 'budget'] = 1941305
    movies_df.loc[923, 'budget'] = 18000000
    movies_df.loc[925, 'budget'] = 5270000
    movies_df.loc[936, 'budget'] = 35000000
    movies_df.loc[937, 'budget'] = 3898383
    movies_df.loc[960, 'budget'] = 25000000
    movies_df.loc[967, 'budget'] = 14807400
    movies_df.loc[988, 'budget'] = 15000000
    movies_df.loc[991, 'budget'] = 2467900
    movies_df.loc[992, 'budget'] = 750000
    movies_df.loc[995, 'budget'] = 24000000
    movies_df.loc[1000, 'budget'] = 700000
    movies_df.loc[929, 'revenue'] = 54000
    movies_df.loc[708, 'revenue'] = 150000
    movies_df.loc[666, 'revenue'] = 12000
    movies_df.loc[665, 'revenue'] = 71000
    movies_df.loc[1012, 'budget'] = 6900000
    movies_df.loc[1024, 'budget'] = 64000
    movies_df.loc[1028, 'budget'] = 12000000
    movies_df.loc[1034, 'budget'] = 20000000
    movies_df.loc[1035, 'budget'] = 850000
    movies_df.loc[1044, 'budget'] = 800000
    movies_df.loc[1054, 'budget'] = 200000
    movies_df.loc[1059, 'budget'] = 3000000
    movies_df.loc[1069, 'budget'] = 7500000
    movies_df.loc[1075, 'budget'] = 4000000
    movies_df.loc[1079, 'budget'] = 1000000
    movies_df.loc[1087, 'budget'] = 5500000
    movies_df.loc[1089, 'budget'] = 1500000
    movies_df.loc[1090, 'budget'] = 1000000
    movies_df.loc[1139, 'budget'] = 37000000
    movies_df.loc[1145, 'revenue'] = 131753
    movies_df.loc[1145, 'budget'] = 15000
    movies_df.loc[1148, 'budget'] = 3500000
    movies_df.loc[1171, 'budget'] = 12000000
    movies_df.loc[1176, 'budget'] = 755979
    movies_df.loc[1195, 'budget'] = 4000000
    movies_df.loc[1214, 'budget'] = 9300000
    movies_df.loc[1222, 'budget'] = 5000000
    movies_df.loc[1241, 'budget'] = 8186604
    movies_df.loc[1253, 'budget'] = 23000000
    movies_df.loc[1258, 'budget'] = 37000000
    movies_df.loc[1259, 'budget'] = 15000000
    movies_df.loc[1262, 'budget'] = 11000000
    movies_df.loc[1290, 'budget'] = 500000
    movies_df.loc[1305, 'budget'] = 11000000
    movies_df.loc[1311, 'budget'] = 4900000
    movies_df.loc[1323, 'budget'] = 3000000
    movies_df.loc[1326, 'budget'] = 2000000
    movies_df.loc[1329, 'budget'] = 4000000
    movies_df.loc[1330, 'budget'] = 2076000
    movies_df.loc[1331, 'budget'] = 12000000
    movies_df.loc[1337, 'budget'] = 10000000
    movies_df.loc[1343, 'budget'] = 1200000
    movies_df.loc[1351, 'budget'] = 20000000
    movies_df.loc[1361, 'budget'] = 1000000
    movies_df.loc[1364, 'budget'] = 15000000
    movies_df.loc[1372, 'budget'] = 4000000
    movies_df.loc[1377, 'budget'] = 650000
    movies_df.loc[1377, 'revenue'] = 5400000
    movies_df.loc[1403, 'budget'] = 10000000
    movies_df.loc[1414, 'budget'] = 20000000
    movies_df.loc[1416, 'budget'] = 7200000
    movies_df.loc[1417, 'budget'] = 30000000
    movies_df.loc[1425, 'budget'] = 12000000
    movies_df.loc[1438, 'budget'] = 465913
    movies_df.loc[1441, 'budget'] = 8400000
    movies_df.loc[1447, 'budget'] = 12000000
    movies_df.loc[1450, 'budget'] = 15000000
    movies_df.loc[1485, 'budget'] = 19000000
    movies_df.loc[1489, 'budget'] = 2500000
    movies_df.loc[1500, 'revenue'] = 680000
    movies_df.loc[1458, 'revenue'] = 300000
    movies_df.loc[1422, 'revenue'] = 8300000
    movies_df.loc[1355, 'revenue'] = 4600000
    movies_df.loc[1345, 'revenue'] = 38000
    movies_df.loc[1311, 'revenue'] = 12800000
    movies_df.loc[1300, 'revenue'] = 83000
    movies_df.loc[1277, 'revenue'] = 2800000
    movies_df.loc[1263, 'revenue'] = 3000000
    movies_df.loc[1241, 'revenue'] = 8300000
    movies_df.loc[1191, 'revenue'] = 7500000
    movies_df.loc[1162, 'revenue'] = 2000000
    movies_df.loc[1142, 'revenue'] = 11000000
    movies_df.loc[1139, 'revenue'] = 30000000
    movies_df.loc[1077, 'revenue'] = 162000
    movies_df.loc[1063, 'revenue'] = 4000
    movies_df.loc[1026, 'revenue'] = 85000
    movies_df.loc[1008, 'revenue'] = 60000

    movies_df.loc[1510, 'budget'] = 3000000
    movies_df.loc[1519, 'budget'] = 500000
    movies_df.loc[1521, 'budget'] = 7000000
    movies_df.loc[1534, 'budget'] = 7000000
    movies_df.loc[1535, 'budget'] = 10000000
    movies_df.loc[1544, 'budget'] = 25000000
    movies_df.loc[1545, 'budget'] = 9800000
    movies_df.loc[1547, 'budget'] = 15000000
    movies_df.loc[1554, 'budget'] = 926073
    movies_df.loc[1555, 'budget'] = 6000000
    movies_df.loc[1575, 'budget'] = 500000
    movies_df.loc[1586, 'budget'] = 20000000
    movies_df.loc[1590, 'budget'] = 1200000
    movies_df.loc[1591, 'budget'] = 4000000
    movies_df.loc[1596, 'budget'] = 50000
    movies_df.loc[1618, 'budget'] = 4600000
    movies_df.loc[1624, 'budget'] = 200000
    movies_df.loc[1628, 'budget'] = 15000000
    movies_df.loc[1646, 'budget'] = 35000000
    movies_df.loc[1684, 'budget'] = 19000000
    movies_df.loc[1697, 'budget'] = 7000000
    movies_df.loc[1698, 'budget'] = 30000000
    movies_df.loc[1701, 'budget'] = 30000000
    movies_df.loc[1702, 'budget'] = 3500000
    movies_df.loc[1707, 'budget'] = 1215000
    movies_df.loc[1725, 'budget'] = 684500
    movies_df.loc[1739, 'budget'] = 50000000
    movies_df.loc[1741, 'budget'] = 18000000
    movies_df.loc[1749, 'budget'] = 10000000
    movies_df.loc[1759, 'budget'] = 45000000
    movies_df.loc[1759, 'revenue'] = 56100000
    movies_df.loc[1766, 'budget'] = 11000000
    movies_df.loc[1767, 'budget'] = 357351
    movies_df.loc[1771, 'budget'] = 10000000
    movies_df.loc[1774, 'budget'] = 30000000
    movies_df.loc[1794, 'budget'] = 1500000
    movies_df.loc[1800, 'budget'] = 4500000
    movies_df.loc[1811, 'budget'] = 6000000
    movies_df.loc[1811, 'revenue'] = 6700000
    movies_df.loc[1830, 'budget'] = 1800000
    movies_df.loc[1843, 'budget'] = 11000000
    movies_df.loc[1853, 'budget'] = 7500000
    movies_df.loc[1813, 'budget'] = 30000000
    movies_df.loc[1826, 'budget'] = 2000000
    movies_df.loc[1835, 'budget'] = 900000
    movies_df.loc[1854, 'budget'] = 8000000
    movies_df.loc[1865, 'budget'] = 50000000
    movies_df.loc[1865, 'revenue'] = 181200000
    movies_df.loc[1868, 'budget'] = 70000000
    movies_df.loc[1882, 'budget'] = 19000000
    movies_df.loc[1883, 'budget'] = 5000000
    movies_df.loc[1895, 'budget'] = 1500000
    movies_df.loc[1900, 'budget'] = 2598922
    movies_df.loc[1905, 'budget'] = 10000000
    movies_df.loc[1906, 'budget'] = 27000000
    movies_df.loc[1907, 'budget'] = 13000000
    movies_df.loc[1910, 'budget'] = 4200000
    movies_df.loc[1913, 'budget'] = 35000000
    movies_df.loc[1916, 'budget'] = 5200000
    movies_df.loc[1919, 'budget'] = 5000000
    movies_df.loc[1932, 'budget'] = 30000000
    movies_df.loc[1933, 'budget'] = 25000000
    movies_df.loc[1942, 'budget'] = 200000
    movies_df.loc[1945, 'budget'] = 5500000
    movies_df.loc[1947, 'budget'] = 3000000
    movies_df.loc[1963, 'budget'] = 30000000
    movies_df.loc[1965, 'budget'] = 1530351
    movies_df.loc[1970, 'budget'] = 1000000
    movies_df.loc[1984, 'budget'] = 60000000
    movies_df.loc[1978, 'revenue'] = 12000
    movies_df.loc[1949, 'revenue'] = 204000
    movies_df.loc[1943, 'revenue'] = 850000
    movies_df.loc[1916, 'revenue'] = 2400000
    movies_df.loc[1879, 'revenue'] = 531000
    movies_df.loc[1875, 'revenue'] = 1000000
    movies_df.loc[1756, 'revenue'] = 342000
    movies_df.loc[1750, 'revenue'] = 4157000
    movies_df.loc[1702, 'revenue'] = 1518000
    movies_df.loc[1691, 'revenue'] = 45000
    movies_df.loc[1649, 'revenue'] = 27000
    movies_df.loc[1575, 'revenue'] = 11000

    ## 09.05.2022 MONDAY
    movies_df.loc[2007, 'budget'] = 31000000
    movies_df.loc[2022, 'budget'] = 2454558
    movies_df.loc[2034, 'budget'] = 9500000
    movies_df.loc[2041, 'budget'] = 4000000
    movies_df.loc[2049, 'budget'] = 3500000
    movies_df.loc[2052, 'budget'] = 6691500
    movies_df.loc[2064, 'budget'] = 10000000
    movies_df.loc[2073, 'budget'] = 10000000
    movies_df.loc[2078, 'budget'] = 18000000
    movies_df.loc[2086, 'budget'] = 5500000
    movies_df.loc[2093, 'budget'] = 1500000
    movies_df.loc[2097, 'budget'] = 50000
    movies_df.loc[2113, 'budget'] = 25000000
    movies_df.loc[2120, 'budget'] = 10000000
    movies_df.loc[2137, 'budget'] = 15000000
    movies_df.loc[2141, 'budget'] = 14000000
    movies_df.loc[2161, 'budget'] = 15000000
    movies_df.loc[2163, 'budget'] = 6500000
    movies_df.loc[2182, 'budget'] = 12000000
    movies_df.loc[2196, 'budget'] = 5500000
    movies_df.loc[2197, 'budget'] = 2500000
    movies_df.loc[2209, 'budget'] = 16000000
    movies_df.loc[2211, 'budget'] = 28300000
    movies_df.loc[2230, 'budget'] = 9000000
    movies_df.loc[2237, 'budget'] = 3100000
    movies_df.loc[2247, 'budget'] = 2400000
    movies_df.loc[2277, 'budget'] = 3700000
    movies_df.loc[2282, 'budget'] = 20000000
    movies_df.loc[2284, 'budget'] = 7000000
    movies_df.loc[2315, 'budget'] = 8000000
    movies_df.loc[2324, 'budget'] = 160299
    movies_df.loc[2333, 'budget'] = 10000000
    movies_df.loc[2337, 'budget'] = 17000000
    movies_df.loc[2338, 'budget'] = 7000000
    movies_df.loc[2346, 'budget'] = 26000000
    movies_df.loc[2349, 'budget'] = 13000000
    movies_df.loc[2360, 'budget'] = 15000000
    movies_df.loc[2367, 'budget'] = 25000000
    movies_df.loc[2368, 'budget'] = 15000000
    movies_df.loc[2372, 'budget'] = 6500000
    movies_df.loc[2384, 'budget'] = 2243432
    movies_df.loc[2386, 'budget'] = 27000000
    movies_df.loc[2387, 'budget'] = 4000000
    movies_df.loc[2394, 'budget'] = 921363
    movies_df.loc[2396, 'budget'] = 8400000
    movies_df.loc[2408, 'budget'] = 8000000
    movies_df.loc[2415, 'budget'] = 12500000
    movies_df.loc[2422, 'budget'] = 5300000
    movies_df.loc[2426, 'budget'] = 17000000
    movies_df.loc[2429, 'budget'] = 26000000
    movies_df.loc[2434, 'budget'] = 5000000
    movies_df.loc[2452, 'budget'] = 7000000
    movies_df.loc[2458, 'budget'] = 20000000
    movies_df.loc[2458, 'revenue'] = 27500000
    movies_df.loc[2459, 'budget'] = 2000
    movies_df.loc[2464, 'budget'] = 1700000
    movies_df.loc[2478, 'budget'] = 1200000
    movies_df.loc[2497, 'budget'] = 40000000
    movies_df.loc[2492, 'budget'] = 100000000
    movies_df.loc[2492, 'revenue'] = 619200000
    movies_df.loc[2475, 'revenue'] = 79000000
    movies_df.loc[2434, 'revenue'] = 5100000
    movies_df.loc[2400, 'revenue'] = 15000000
    movies_df.loc[2385, 'revenue'] = 700000
    movies_df.loc[2384, 'revenue'] = 35300000
    movies_df.loc[2375, 'revenue'] = 160000
    movies_df.loc[2331, 'revenue'] = 353000
    movies_df.loc[2324, 'revenue'] = 40000
    movies_df.loc[2091, 'revenue'] = 1700000
    movies_df.loc[2504, 'budget'] = 17000000
    movies_df.loc[2505, 'budget'] = 8500000
    movies_df.loc[2512, 'budget'] = 8000000
    movies_df.loc[2537, 'budget'] = 9000000
    movies_df.loc[2540, 'budget'] = 8500000
    movies_df.loc[2544, 'budget'] = 30000000
    movies_df.loc[2551, 'budget'] = 30000000
    movies_df.loc[2555, 'budget'] = 15000000
    movies_df.loc[2557, 'budget'] = 3400000
    movies_df.loc[2561, 'budget'] = 12000000
    movies_df.loc[2569, 'budget'] = 9000000
    movies_df.loc[2578, 'budget'] = 1500
    movies_df.loc[2584, 'budget'] = 20000000
    movies_df.loc[2586, 'budget'] = 4000000
    movies_df.loc[2599, 'revenue'] = 17200000
    movies_df.loc[2599, 'budget'] = 16500000
    movies_df.loc[2601, 'budget'] = 3000000
    movies_df.loc[2603, 'budget'] = 5000000
    movies_df.loc[2606, 'budget'] = 25000000
    movies_df.loc[2616, 'budget'] = 6000000
    movies_df.loc[2626, 'budget'] = 1000000
    movies_df.loc[2626, 'revenue'] = 51500000
    movies_df.loc[2641, 'budget'] = 12000000
    movies_df.loc[2642, 'budget'] = 10000000
    movies_df.loc[2647, 'budget'] = 1000000
    movies_df.loc[2655, 'budget'] = 10500000
    movies_df.loc[2659, 'budget'] = 15000000
    movies_df.loc[2666, 'budget'] = 11000000
    movies_df.loc[2672, 'budget'] = 25000000
    movies_df.loc[2685, 'budget'] = 5500000
    movies_df.loc[2695, 'budget'] = 8000000
    movies_df.loc[2700, 'budget'] = 3000000
    movies_df.loc[2709, 'budget'] = 9500000
    movies_df.loc[2719, 'budget'] = 1100000
    movies_df.loc[2737, 'budget'] = 12000000
    movies_df.loc[2746, 'budget'] = 40000000
    movies_df.loc[2750, 'budget'] = 3000000
    movies_df.loc[2752, 'budget'] = 10000000
    movies_df.loc[2755, 'budget'] = 5500000
    movies_df.loc[2758, 'budget'] = 270000
    movies_df.loc[2760, 'budget'] = 85000000
    movies_df.loc[2766, 'budget'] = 5000000
    movies_df.loc[2772, 'budget'] = 9000000
    movies_df.loc[2786, 'budget'] = 20000000
    movies_df.loc[2802, 'budget'] = 2500000
    movies_df.loc[2804, 'budget'] = 12000000
    movies_df.loc[2811, 'budget'] = 12000000
    movies_df.loc[2818, 'budget'] = 20000000
    movies_df.loc[2825, 'budget'] = 630000
    movies_df.loc[2828, 'budget'] = 11000000
    movies_df.loc[2834, 'budget'] = 32000000
    movies_df.loc[2841, 'budget'] = 11262673
    movies_df.loc[2848, 'budget'] = 9981144
    movies_df.loc[2853, 'budget'] = 12000000
    movies_df.loc[2854, 'budget'] = 19000000
    movies_df.loc[2875, 'budget'] = 25000
    movies_df.loc[2880, 'budget'] = 1300000
    movies_df.loc[2890, 'budget'] = 9000000
    movies_df.loc[2893, 'budget'] = 2463580
    movies_df.loc[2897, 'budget'] = 1000000
    movies_df.loc[2903, 'budget'] = 350000
    movies_df.loc[2908, 'budget'] = 25000000
    movies_df.loc[2918, 'budget'] = 20000000
    movies_df.loc[2919, 'budget'] = 13500000
    movies_df.loc[2921, 'budget'] = 35000000
    movies_df.loc[2922, 'budget'] = 5000000
    movies_df.loc[2948, 'budget'] = 8000000
    movies_df.loc[2957, 'budget'] = 8000000
    movies_df.loc[2974, 'budget'] = 13000000
    movies_df.loc[2979, 'budget'] = 9000000
    movies_df.loc[2991, 'budget'] = 10000000
    movies_df.loc[2995, 'budget'] = 18000000
    movies_df.loc[2996, 'budget'] = 15000000
    movies_df.loc[2875, 'revenue'] = 25000
    movies_df.loc[2865, 'revenue'] = 100000
    movies_df.loc[2846, 'revenue'] = 33000
    movies_df.loc[2811, 'revenue'] = 500000
    movies_df.loc[2760, 'revenue'] = 130000
    movies_df.loc[2737, 'revenue'] = 1100000
    movies_df.loc[2626, 'revenue'] = 500000
    movies_df.loc[2601, 'revenue'] = 2500000
    movies_df.loc[2583, 'revenue'] = 800000
    movies_df.loc[2578, 'revenue'] = 15000

    movies_df.loc[6, 'production_countries'] = 'India'
    movies_df.loc[8, 'production_countries'] = 'United States of America'
    movies_df.loc[39, 'production_countries'] = 'United States of America'
    movies_df.loc[260, 'production_countries'] = 'India'
    movies_df.loc[270, 'production_countries'] = 'United States of America'
    movies_df.loc[446, 'production_countries'] = 'United States of America'
    movies_df.loc[458, 'production_countries'] = 'United States of America'
    movies_df.loc[471, 'production_countries'] = 'United States of America'
    movies_df.loc[499, 'production_countries'] = 'United States of America'
    movies_df.loc[610, 'production_countries'] = 'United States of America'
    movies_df.loc[630, 'production_countries'] = 'United States of America'
    movies_df.loc[637, 'production_countries'] = 'United States of America'
    movies_df.loc[653, 'production_countries'] = 'United Kingdom'
    movies_df.loc[665, 'production_countries'] = 'United States of America'
    movies_df.loc[738, 'production_countries'] = 'United States of America'
    movies_df.loc[753, 'production_countries'] = 'United States of America'
    movies_df.loc[830, 'production_countries'] = 'United States of America'
    movies_df.loc[980, 'production_countries'] = 'United States of America'
    movies_df.loc[1087, 'production_countries'] = 'United States of America'
    movies_df.loc[1111, 'production_countries'] = 'United States of America'
    movies_df.loc[1241, 'production_countries'] = 'India'
    movies_df.loc[1254, 'production_countries'] = 'United States of America'
    movies_df.loc[1334, 'production_countries'] = 'United States of America'
    movies_df.loc[1336, 'production_countries'] = 'Russia'
    movies_df.loc[1345, 'production_countries'] = 'United States of America'
    movies_df.loc[1484, 'production_countries'] = 'United States of America'
    movies_df.loc[1504, 'production_countries'] = 'United States of America'
    movies_df.loc[1623, 'production_countries'] = 'United States of America'
    movies_df.loc[1649, 'production_countries'] = 'United States of America'
    movies_df.loc[1684, 'production_countries'] = 'United States of America'
    movies_df.loc[1758, 'production_countries'] = 'Vietnam'
    movies_df.loc[1772, 'production_countries'] = 'South Korea'
    movies_df.loc[1786, 'production_countries'] = 'United States of America'
    movies_df.loc[1892, 'production_countries'] = 'United States of America'
    movies_df.loc[1924, 'production_countries'] = 'United States of America'
    movies_df.loc[2011, 'production_countries'] = 'India'
    movies_df.loc[2014, 'production_countries'] = 'France'
    movies_df.loc[2130, 'production_countries'] = 'Italy'
    movies_df.loc[2282, 'production_countries'] = 'United States of America'
    movies_df.loc[2296, 'production_countries'] = 'France'
    movies_df.loc[2343, 'production_countries'] = 'Turkey'
    movies_df.loc[2428, 'production_countries'] = 'United States of America'
    movies_df.loc[2445, 'production_countries'] = 'United States of America'
    movies_df.loc[2505, 'production_countries'] = 'United States of America'
    movies_df.loc[2511, 'production_countries'] = 'Finland'
    movies_df.loc[2687, 'production_countries'] = 'Russia'
    movies_df.loc[2733, 'production_countries'] = 'United States of America'
    movies_df.loc[2875, 'production_countries'] = 'United States of America'
    movies_df.loc[2887, 'production_countries'] = 'United States of America'
    movies_df.loc[2920, 'production_countries'] = 'United States of America'
    movies_df.loc[2922, 'production_countries'] = 'United States of America'
    movies_df.loc[2937, 'production_countries'] = 'Italy'
    movies_df.loc[2948, 'production_countries'] = 'United States of America'
    movies_df.loc[2991, 'production_countries'] = 'United States of America'

    return movies_df
