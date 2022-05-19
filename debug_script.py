from sklearn.model_selection import train_test_split

from model_utils import *
from utils import clean_data, date_features, eval_dict_columns
from utils import load_data, join_macroeconomics, categorical_cut


def processing():
    movies_df, cci_df, cpi_df, gdp_df, rir_df, uer_df, eacc_df, rotwcc_df = load_data()

    movies_df = clean_data(movies_df)
    movies_df = date_features(movies_df)
    movies_df = eval_dict_columns(movies_df)

    movies_df = movies_df[movies_df.year >= 1961]
    budget_median_in_buckets = movies_df.groupby(['year']).budget.transform('median')
    movies_df.loc[movies_df['budget'] == 0, 'budget'] = None
    movies_df.budget = movies_df.budget.fillna(budget_median_in_buckets)

    # movies_df = cast_crew_features(movies_df)
    cutoff_thresholds = {
        'genres': 0,
        'production_companies': 30,
        'production_countries': 0,
        'spoken_languages': 100,
        'Keywords': 30,
        'cast': 0,
        'crew': 0,
    }
    movies_df, results_summary_df = categorical_cut(movies_df, cutoff_thresholds)
    movies_df = join_macroeconomics(movies_df, cci_df, cpi_df, gdp_df, rir_df, uer_df, eacc_df, rotwcc_df)


def model_training():
    X, y = load_X_y()
    y = categorical_threasholding_y(y, multiclass_percentiles=[0.33, 0.66])
    cat = pd.Categorical(y, ordered=True, categories=['Unsuccesful', 'Neutral', 'Succesful'])
    y = pd.Series(cat.codes)
    y_labels = cat.categories
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
    X_train_sel, X_test_sel = boruta_feature_selection(X_train, X_test, y_train, max_iter=2)
    cv_results = run_grid_search(X_train_sel, y_train, n_iter=2)
    evaluate_best_models(cv_results, X_train_sel, y_train, X_test_sel, y_test, labels=[0, 1, 2], target_names=y_labels)


if __name__ == "__main__":
    # processing()
    model_training()
