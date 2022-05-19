import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from settings import MACROECONOMIC_FEATURE_NAMES, param_grid, drop_transformer


def load_X_y():
    df = pd.read_csv('data/processed_data.csv', index_col=0)
    df = df[df.budget > 0]
    df = df[df.runtime > 0]
    # df['roi'] = (df.revenue-df.budget)/df.budget
    df = df.drop(columns=['production_countries', 'iso_3166_1_alpha_2', 'country_category', 'revenue'])
    df = df.dropna()
    y = df.pop('roi')
    X = df.copy()

    df_columns = X.columns
    return X, y


def calculate_threshold(y, percentile):
    threshold = y.quantile(1 - percentile)
    return threshold


def plot_roi_thresholding(y, threshold):
    y[y < 10].hist(bins=100)
    plt.axvline(threshold, c='r')
    plt.savefig("images/roi_thresholding.svg")


def categorical_threasholding_y(y, single_percentile=None, multiclass_percentiles=None):
    if single_percentile is not None:
        threshold = calculate_threshold(y, single_percentile)
        plot_roi_thresholding(y, threshold)
        y_categorical = (y > threshold).astype(int)
    elif multiclass_percentiles is not None:
        assert len(multiclass_percentiles) == 2, "len(multiclass_percentiles) should be 2"
        threshold = [calculate_threshold(y, percentile) for percentile in sorted(multiclass_percentiles)]
        y_categorical = y.copy()
        y_categorical[y > threshold[0]] = "Succesful"
        y_categorical[(y > threshold[1]) & (y <= threshold[0])] = "Neutral"
        y_categorical[y <= threshold[1]] = "Unsuccesful"
        y_categorical
    else:
        assert False, "set at least one of [single_percentile, multiclass_percentile] to not None"
    return y_categorical


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


def boruta_feature_selection(X_train, X_test, y_train, max_iter=100):
    X_train_no_macro = X_train.drop(columns=MACROECONOMIC_FEATURE_NAMES)
    forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    forest.fit(X_train_no_macro, y_train)
    feat_selector = BorutaPy(forest, n_estimators='auto', verbose=1, random_state=1, max_iter=max_iter)
    feat_selector.fit(X_train_no_macro.values, y_train.values)

    selected_columns = list(X_train_no_macro.columns[feat_selector.support_]) + MACROECONOMIC_FEATURE_NAMES

    X_train_sel = X_train[selected_columns]
    X_test_sel = X_test[selected_columns]

    return X_train_sel, X_test_sel


def run_grid_search(X_train_sel, y_train, n_iter=20):
    pipeline = Pipeline([("marcro_drop", None), ("scaler", StandardScaler()), ("clf", None)])
    search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, verbose=2)
    #search = RandomizedSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, verbose=2, n_iter=n_iter)  # tylko na testy
    search.fit(X_train_sel, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results.dropna(subset=['param_marcro_drop'])
    cv_results['param_clf_name'] = cv_results.param_clf.apply(lambda x: str(x).split('(')[0])
    _unique_macro_drop_names = cv_results.param_marcro_drop.unique().astype(str)
    _unique_macro_drop_names.sort()
    _replace_dict = {k: v for k, v in zip(_unique_macro_drop_names, ['without_macro', 'with_macro'])}
    cv_results['marcro_drop_name'] = cv_results.param_marcro_drop.astype(str).replace(_replace_dict)
    cv_results = cv_results.drop(
        columns=['params', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'split0_test_score',
                 'split1_test_score', 'split2_test_score']).sort_values('rank_test_score')
    return cv_results

def run_grid_search2(X_train_sel, y_train, n_iter=30):
    pipeline = Pipeline([("marcro_drop", None), ("scaler", StandardScaler()), ("clf", None)])
    search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, verbose=2)
    #search = RandomizedSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, verbose=2, n_iter=n_iter)  # tylko na testy
    search.fit(X_train_sel, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results.dropna(subset=['param_marcro_drop'])
    cv_results['param_clf_name'] = cv_results.param_clf.apply(lambda x: str(x).split('(')[0])
    _unique_macro_drop_names = cv_results.param_marcro_drop.unique().astype(str)
    _unique_macro_drop_names.sort()
    _replace_dict = {k: v for k, v in zip(_unique_macro_drop_names, ['without_macro', 'with_macro'])}
    cv_results['marcro_drop_name'] = cv_results.param_marcro_drop.astype(str).replace(_replace_dict)
    cv_results = cv_results.drop(
        columns=['params', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'split0_test_score',
                 'split1_test_score', 'split2_test_score']).sort_values('rank_test_score')
    return cv_results



def compare_models_plot(cv_results):
    fig, ax = plt.subplots(figsize=(12, 5))
    # ax = sns.violinplot(x="param_clf_name", y="mean_test_score", hue="marcro_drop_name", palette="Set2", data=cv_results, inner=None, dodge=True, alpha=0.3)
    sns.swarmplot(data=cv_results, y="mean_test_score", x="param_clf_name", hue="marcro_drop_name", palette="Set2",
                  dodge=True, ax=ax)
    plt.savefig('images/model_compare_grid_search.svg')


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(selected_model, selected_model_name, X_test_sel, y_test, labels=[0, 1],
                   target_names=['Non Succesful', 'Succesful']):
    y_pred = selected_model.predict(X_test_sel)

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.title(selected_model_name)
    plt.show()
    print(classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names))
    

def evaluate_model2(selected_model, selected_model_name, X_test_sel, y_test, labels=[0, 1],
                   target_names=['Non Succesful', 'Succesful']):
    y_pred = selected_model.predict(X_test_sel)
    
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(round(auc,3)))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.title(selected_model_name)
    plt.show()


def evaluate_best_models(cv_results, X_train_sel, y_train, X_test_sel, y_test, labels=[0, 1],
                         target_names=['Non Succesful', 'Succesful']):
    best_models_results = (
        cv_results.sort_values("rank_test_score")
            .groupby(["param_clf_name", "param_marcro_drop"])
            .head(1)
    )
    for param_clf_name in best_models_results.param_clf_name.unique():
        model_params_df = (
            best_models_results[best_models_results.param_clf_name == param_clf_name]
                .dropna(axis=1)
                .drop(columns=["mean_test_score", "std_test_score", "rank_test_score"], errors=['ignore'])
        )
        for macro_drop_name in model_params_df.marcro_drop_name.unique():
            model_params_sel_macro_df = model_params_df[
                model_params_df.marcro_drop_name == macro_drop_name
                ]
            if model_params_sel_macro_df.empty:
                continue
            model = model_params_sel_macro_df.param_clf.item()
            model_name = model_params_sel_macro_df.param_clf_name.item()
            macro_drop = (
                drop_transformer if macro_drop_name == "without_macro" else "passthrough"
            )
            pipeline = Pipeline(
                [("marcro_drop", macro_drop), ("scaler", StandardScaler()), ("clf", model)]
            )
            pipeline.fit(X_train_sel, y_train)
            pipeline.predict(X_test_sel)
            evaluate_model(
                selected_model_name=f"{model_name}/{macro_drop_name}",
                selected_model=pipeline,
                X_test_sel=X_test_sel,
                y_test=y_test,
                labels=labels,
                target_names=target_names
            )

            
            
def evaluate_best_models2(cv_results, X_train_sel, y_train, X_test_sel, y_test, labels=[0, 1],
                         target_names=['Non Succesful', 'Succesful']):
    best_models_results = (
        cv_results.sort_values("rank_test_score")
            .groupby(["param_clf_name", "param_marcro_drop"])
            .head(1)
    )
    for param_clf_name in best_models_results.param_clf_name.unique():
        model_params_df = (
            best_models_results[best_models_results.param_clf_name == param_clf_name]
                .dropna(axis=1)
                .drop(columns=["mean_test_score", "std_test_score", "rank_test_score"], errors=['ignore'])
        )
        for macro_drop_name in model_params_df.marcro_drop_name.unique():
            model_params_sel_macro_df = model_params_df[
                model_params_df.marcro_drop_name == macro_drop_name
                ]
            if model_params_sel_macro_df.empty:
                continue
            model = model_params_sel_macro_df.param_clf.item()
            model_name = model_params_sel_macro_df.param_clf_name.item()
            macro_drop = (
                drop_transformer if macro_drop_name == "without_macro" else "passthrough"
            )
            pipeline = Pipeline(
                [("marcro_drop", macro_drop), ("scaler", StandardScaler()), ("clf", model)]
            )
            pipeline.fit(X_train_sel, y_train)
            pipeline.predict(X_test_sel)
            evaluate_model2(
                selected_model_name=f"{model_name}/{macro_drop_name}",
                selected_model=pipeline,
                X_test_sel=X_test_sel,
                y_test=y_test,
                labels=labels,
                target_names=target_names
            )

# def plot_multiclass_roc(selected_model, selected_model_name, X_test_sel, y_test, n_classes):
#     y_pred = selected_model.predict(X_test_sel)

#     # structures
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()

#     # calculate dummies once
#     y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_pred[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])

#     # roc for each class
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.plot([0, 1], [0, 1], 'k--')
#     ax.set_xlim([0.0, 1.0])
#     ax.set_ylim([0.0, 1.05])
#     ax.set_xlabel('False Positive Rate')
#     ax.set_ylabel('True Positive Rate')
#     ax.title(selected_model_name)
#     for i in range(n_classes):
#         ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
#     ax.legend(loc="best")
#     ax.grid(alpha=.4)
#     sns.despine()
#     plt.show()




# def evaluate_best_models_roc_curve(cv_results, X_train_sel, y_train, X_test_sel, y_test, labels=[0, 1],
#                          target_names=['Non Succesful', 'Succesful']):
#     best_models_results = (
#         cv_results.sort_values("rank_test_score")
#             .groupby(["param_clf_name", "param_marcro_drop"])
#             .head(1)
#     )
#     for param_clf_name in best_models_results.param_clf_name.unique():
#         model_params_df = (
#             best_models_results[best_models_results.param_clf_name == param_clf_name]
#                 .dropna(axis=1)
#                 .drop(columns=["mean_test_score", "std_test_score", "rank_test_score"], errors=['ignore'])
#         )
#         for macro_drop_name in model_params_df.marcro_drop_name.unique():
#             model_params_sel_macro_df = model_params_df[
#                 model_params_df.marcro_drop_name == macro_drop_name
#                 ]
#             if model_params_sel_macro_df.empty:
#                 continue
#             model = model_params_sel_macro_df.param_clf.item()
#             model_name = model_params_sel_macro_df.param_clf_name.item()
#             macro_drop = (
#                 drop_transformer if macro_drop_name == "without_macro" else "passthrough"
#             )
#             pipeline = Pipeline(
#                 [("marcro_drop", macro_drop), ("scaler", StandardScaler()), ("clf", model)]
#             )
#             pipeline.fit(X_train_sel, y_train)
#             pipeline.predict(X_test_sel)
#             plot_multiclass_roc(
#                 selected_model_name=f"{model_name}/{macro_drop_name}",
#                 selected_model=pipeline, 
#                 X_test_sel=X_test_sel, 
#                 y_test=y_test, 
#                 n_classes
#             )
        