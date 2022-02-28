import pandas as pd
import numpy as np
import unidecode
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from scipy import interp

# Sklearn imports

from sklearn.model_selection import (
        train_test_split,
        learning_curve,
        StratifiedKFold,
        cross_val_score
        )

from sklearn.metrics import (
        roc_auc_score,
        roc_curve,
        # precision_recall_curve,
        confusion_matrix,
        auc
        )

CORR_THRESH = 0.95
TRAIN_SIZES = np.linspace(.2, 1.0, 10)
SEED = 123

def plots_freq_target(data,target,vars_cat=np.nan,vars_num=np.nan):
        
    if pd.notnull([vars_cat]).any() and pd.notnull([vars_num]).any():
        
        with PdfPages('target_'+target+'_variables_frequency.pdf') as pdf_pages:
            for col in data[vars_cat].columns:
                figu = plt.figure(col)
                ax = sns.countplot(x=col,hue=target,data=data[[col,target]].fillna('missing').astype('category'))
                
                bars = ax.patches
                half = int(len(bars)/2)
                left_bars = bars[:half]
                right_bars = bars[half:]
                
                for left, right in zip(left_bars, right_bars):
                    height_l = float(np.where(pd.isnull(left.get_height()),0,left.get_height()))
                    height_r = float(np.where(pd.isnull(right.get_height()),0,right.get_height()))
                    total = height_l + height_r
                    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
                    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
                    
                plt.title("Distribucion "+col)
                plt.legend(labels=['Target 0','Target 1'],prop={'size': 12},loc='upper right')
                plt.ylabel('Frecuencia')
                pdf_pages.savefig(figu)
                plt.close()
                print(col)                          
        
            for col in data[vars_num].columns:
                y0 = data.loc[data[target]==0,col].replace([np.inf, -np.inf], np.nan).dropna()#.apply(np.log1p)
                y1 = data.loc[data[target]==1,col].replace([np.inf, -np.inf], np.nan).dropna()#.apply(np.log1p)
            # Remove below 10% and over 90% quantiles to discard outliers
                y0 = y0[y0.between(y0.quantile(0.05), y0.quantile(0.95))]                    
                y1 = y1[y1.between(y1.quantile(0.05), y1.quantile(0.95))]                    
                figu = plt.figure(col)
                sns.distplot(y0.dropna(),kde=True,norm_hist=True,kde_kws={"shade": False})    
                sns.distplot(y1.dropna(),kde=True,norm_hist=True,kde_kws={"shade": False})    

                plt.title('Distribucion '+col)
                plt.legend(labels=['Target 0','Target 1'],prop={'size': 12},loc='upper right')
                plt.ylabel('Frecuencia')
                pdf_pages.savefig(figu)
                plt.close()
                print(col)
                
    else:
        
        with PdfPages('target_'+target+'_variables_frequency.pdf') as pdf_pages:
            if pd.notnull(data.columns[data.dtypes=='object']).any():
                for col in data.columns[data.dtypes=='object']:
                    figu = plt.figure(col)
                    ax = sns.countplot(x=col,hue=target,data=data[[col,target]].fillna('missing').astype('category'))
                 
                    bars = ax.patches
                    half = int(len(bars)/2)
                    left_bars = bars[:half]
                    right_bars = bars[half:]
                    
                    for left, right in zip(left_bars, right_bars):
                        height_l = float(np.where(pd.isnull(left.get_height()),0,left.get_height()))
                        height_r = float(np.where(pd.isnull(right.get_height()),0,right.get_height()))
                        total = height_l + height_r
                        ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
                        ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
                    
                    plt.title("Distribucion "+col)
                    plt.legend(labels=['Target 0','Target 1'],prop={'size': 12},loc='upper right')
                    plt.ylabel('Frecuencia')
                    pdf_pages.savefig(figu)
                    plt.close()
                    print(col)                       
        
            if pd.notnull(data.columns[data.dtypes!='object']).any():
                for col in data.columns[data.dtypes!='object']:
                    y0 = data.loc[data[target]==0,col].replace([np.inf, -np.inf], np.nan).dropna()#.apply(np.log1p)
                    y1 = data.loc[data[target]==1,col].replace([np.inf, -np.inf], np.nan).dropna()#.apply(np.log1p)
                    # Remove below 10% and over 90% quantiles to discard outliers
                    y0 = y0[y0.between(y0.quantile(0.05), y0.quantile(0.95))]                    
                    y1 = y1[y1.between(y1.quantile(0.05), y1.quantile(0.95))]                    
                    figu = plt.figure(col)
                    sns.distplot(y0.dropna(),kde=True,norm_hist=True,kde_kws={"shade": False})    
                    sns.distplot(y1.dropna(),kde=True,norm_hist=True,kde_kws={"shade": False})    

                    plt.title('Distribucion '+col)
                    plt.legend(labels=['Target 0','Target 1'],prop={'size': 12},loc='upper right')
                    plt.ylabel('Frecuencia')
                    pdf_pages.savefig(figu)
                    plt.close()
                    print(col)
                    
# SECTION 2: Auxiliary functions

def plot_learning_curve(
        classifier,
        title,
        X,
        Y,
        cv,
        train_sizes=TRAIN_SIZES
        ):
    '''
    This function generates the learning plot for the given 'classifier' for
    data X, y using cross-validation as specified in 'cv'. The evaluation
    metric is AUC-ROC.

    The process splits the training data in sets of increasing size in order to
    visualize how adding data to the model improves its performance.
    '''

    # Calculation of learning curve

    train_sizes, train_scores, test_scores = learning_curve(
        classifier,
        X,
        Y,
        cv=cv,
        train_sizes=train_sizes,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=SEED
        )

    # Calculation of mean, std for all folds

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot train/test curve and regions representing std

    plt.figure()

    plt.plot(
            train_sizes,
            train_scores_mean,
            'o-',
            color="r",
            label="Training score"
            )

    plt.plot(
            train_sizes,
            test_scores_mean,
            'o-',
            color="g",
            label="CV score"
            )

    plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r"
            )

    plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g"
            )

    # Plot properties

    plt.title(title)
    plt.ylim([0.6, 1.01])
    plt.xlabel('Training examples')
    plt.ylabel('AUC Score')
    plt.grid()
    plt.legend(loc=4)


def plot_confusion_matrix(confusion_matrix, title):
    '''
    Plots the confusion matrix, which is provided by the output of the function
    sklearn.metrics.confusion_matrix.
    '''

    plt.figure()
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt='d',
                linewidths=.5,
                cmap='winter',
                annot_kws={'size': 16},
                alpha=0.8
                )
    plt.xlabel('Predicted class')
    plt.ylabel('Actual class')
    plt.title(title)
    plt.show()


def plot_ROC(fpr, tpr, title, auc):
    '''
    Plots ROC curve.
    '''

    plt.figure()
    plt.plot(fpr, tpr, label='AUC-ROC: ' + str(np.round(auc, 4)))

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()


def plot_ROC_kfolds(classifier, X_train, y_train, kfold, title):
    '''
    Plots ROC curves for all kfolds and calculates the average curve.
    '''

    mean_fpr = np.linspace(0, 1, 100)

    #fig = plt.figure(figsize=(6*2.13, 4*2.13))
    fig = plt.figure(figsize=(6, 4))

    tprs = []
    # aucs = []

    for train, test in kfold.split(X_train, y_train):

        probas = classifier.fit(
                X_train.iloc[train, :], y_train.iloc[train]).predict_proba(
                        X_train.iloc[test, :]
                        )

        fpr, tpr, thresholds = roc_curve(
                y_train.iloc[test],
                probas[:, 1], pos_label=1
                )

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # roc_auc=auc(fpr, tpr)
        # threaucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1)

    plt.plot([0, 1], [0, 1], linestyle='--')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(
            mean_fpr,
            mean_tpr,
            'k--',
            label='Mean AUC (%0.4f)' % mean_auc,
            lw=2
            )

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend(loc=4)
    plt.grid()
    fig.tight_layout()


def remove_correl_feat(df, thresh):
    '''
    This function receives a pandas dataframe "df" and a float (e.g. 0.8),
    returning a reduced version of df where all variables that satisfy
    correlation > thresh, are removed.
    '''

    # Correlation matrix

    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix

    upper = corr_matrix.where(np.triu(
            np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than thresh

    to_drop = [
            column for column in upper.columns if any(upper[column] > thresh)
            ]

    # Return reduced dataframe

    print('Features removed due to high pair-wise correlation:')

    for feat_name in to_drop:

        print(feat_name)

    return df.drop(to_drop, axis=1)


def plot_corr_heatmap(df):
    '''
    This functions receives a dataframe "df" and plots a heatmap of the
    correlation matrix.
    '''

    corr_mat = df.corr()

    plt.figure(figsize=(18, 12))

    sns.heatmap(
            corr_mat,
            vmax=1,
            vmin=-1,
            square=False,
            annot=True,
            cmap="PiYG"
            )

    plt.tight_layout()

def prepare_datasets(dataset, na_thresh):
    '''
    '''

    # Make a copy of the input df

    df = dataset.copy()

    # Remove correlated features

    #df = remove_correl_feat(df, CORR_THRESH)

    # Drop rows that have high missing data

    df.dropna(axis=0, thresh=int(na_thresh * df.shape[1]), inplace=True)

    # Specify categories for categorical variables (based on observed values)

    df['activo_pasivo'] = pd.Categorical(
            df.activo_pasivo,
            ['missing','activo_pasivo', 'pasivo', 'activo', 'otro']
            )
    
    df['mono_multi'] = pd.Categorical(
                df.mono_multi,
                ['otro_producto', 'monoproducto', 'multiproducto']
                )
    
    df['lealtad'] = pd.Categorical(
                df.lealtad,
                ['volatil', 'ocasional', 'prefiere', 'leal']
                )
    
    df['rentab_categ'] = pd.Categorical(
                df.rentab_categ,
                ['poco rentable', 'rentable', 'no rentable', 'muy rentable']
                )
    
    df['predict_python'] = pd.Categorical(
                df.predict_python,
                ['cl-8','cl-7','cl-6','cl-5','cl-4','cl-3', 'cl-2', 'cl-1']
                )
    
#    df['seg_cliente_anual'] = pd.Categorical(
#                df.seg_cliente_anual,
#                ['missing', 'grandes', 'pymes']
#                )

    df['segmento'] = pd.Categorical(
                df.segmento,
                ['pyme', 'empresarial']
                )
    
    df['emergente_anual'] = pd.Categorical(
                df.emergente_anual,
                ['missing', 'emergente', 'no_emergente', 'no_identificado']
                )
    
    df['capa_cliente_anual'] = pd.Categorical(
                df.capa_cliente_anual,
                ['missing', 'oro', 'bronce', 'plata']
                )
    
    df['min_cal_anual'] = pd.Categorical(
                df.min_cal_anual,
                ['missing', 'A', 'B', 'C', 'D', 'E', 'K']
                )
    
    df['min_cal_trim_ant'] = pd.Categorical(
                df.min_cal_trim_ant,
                ['missing', 'A', 'B', 'C', 'D', 'E', 'K']
                )
    
    df['max_cal_anual'] = pd.Categorical(
                df.max_cal_anual,
                ['missing', 'A', 'B', 'C', 'D', 'E', 'K']
                )
    
    df['max_cal_trim_ant'] = pd.Categorical(
                df.max_cal_trim_ant,
                ['missing', 'A', 'B', 'C', 'D', 'E', 'K']
                )

    df['segmento_cluster'] = pd.Categorical(
                df.segmento_cluster,
                ['pyme_cl-8', 'pyme_cl-1', 'empresarial_cl-1', 'pyme_cl-5', 'pyme_cl-4',
                 'pyme_cl-3', 'pyme_cl-7', 'pyme_cl-6',
                 'empresarial_cl-3', 'pyme_cl-2', 'empresarial_cl-2']
                )
    
    # Separate in two models for 'activos' and 'pasivos' based on either:
    
    # 'x' represents the training set and 'y' the target variable
    
    vars_cat = ['segmento_cluster','activo_pasivo','lealtad','mono_multi','predict_python','rentab_categ','segmento',
                'emergente_anual','capa_cliente_anual','min_cal_anual','min_cal_trim_ant','max_cal_anual','max_cal_trim_ant']

    #vars_activo = ['leasing_bin','cartera_ordinaria_bin','tarjeta_credito_bin',
    #           'creditos_tesoreria_bin','unidirecto_bin','factoring_bin']

    #vars_pasivo = ['cuenta_corriente_bin','cuenta_ahorros_bin']
    
    vars_tesoreria = ['tenencia_FWD', 'tenencia_FWD_t_3', 'tenencia_FWD_t_6',
                      'tenencia_SPOT', 'tenencia_SPOT_t_3', 'tenencia_SPOT_t_6']
    
    #vars_maybe = ['bienes_consumo_intermedio','caf','cartera_fomento','cartera_fomento_autos_&_bc','cartera_preoperativa','cdt',
    #              'credito_empleados','credito_hipotecario','credito_rotativo','deudores_en_me','fomento_en_me','libranza','motos',
    #              'prestamo_personal','rec_congelado_rem','recaudo_de_impuestos','sobregiro','vehiculos','bienes_consumo_intermedio_bin',
    #              'caf_bin','cartera_fomento_bin','cartera_fomento_autos_&_bc_bin','cartera_preoperativa_bin','cdt_bin',
    #              'credito_empleados_bin','credito_hipotecario_bin','credito_rotativo_bin','deudores_en_me_bin','fomento_en_me_bin',
    #              'libranza_bin','motos_bin','prestamo_personal_bin','rec_congelado_rem_bin','recaudo_de_impuestos_bin','sobregiro_bin',
    #              'vehiculos_bin']

    vars_maybe = []#'monto_usd_mean_t_6','utilidad_mean_t_6']
    
    vars_num = [col for col in df.columns if (col not in vars_cat) and ('str_identificacion' not in col) and
                                             (col not in vars_tesoreria)]

    x_tesoreria = df[[col for col in df.columns if ((col in vars_num) or (col in vars_cat)) and (col not in vars_maybe)]]
    #x_activo = df[[col for col in df.columns if ((col in vars_num) or (col in vars_cat)) and (col not in vars_maybe)]]
    #x_pasivo = df[[col for col in df.columns if ((col in vars_num) or (col in vars_cat)) and (col not in vars_maybe)]]

    y_tesoreria = df[vars_tesoreria]
    #y_activo = df[vars_activo]
    #y_pasivo = df[vars_pasivo]

    # Get dummies

    x_tesoreria = pd.get_dummies(
            x_tesoreria,
            columns=x_tesoreria.columns[
                    np.where(x_tesoreria.dtypes == 'category')[0]
                    ],
            drop_first=True
            )

    #x_pasivo = pd.get_dummies(
    #        x_pasivo,
    #        columns=x_pasivo.columns[
    #                np.where(x_pasivo.dtypes == 'category')[0]
    #                ],
    #        drop_first=True
    #        )

    # Return df for both models and target variables

    #return x_activo, x_pasivo, y_activo, y_pasivo
    return x_tesoreria, y_tesoreria

def prepare_datasets_sinseg(dataset, na_thresh):
    '''
    '''

    # Make a copy of the input df

    df = dataset.copy()

    # Remove correlated features

    #df = remove_correl_feat(df, CORR_THRESH)

    # Drop rows that have high missing data

    df.dropna(axis=0, thresh=int(na_thresh * df.shape[1]), inplace=True)

    # Specify categories for categorical variables (based on observed values)

    #df['activo_pasivo'] = pd.Categorical(
    #        df.activo_pasivo,
    #        ['activo_pasivo', 'pasivo', 'activo', 'otro']
    #        )
    #
    #df['mono_multi'] = pd.Categorical(
    #            df.mono_multi,
    #            ['monoproducto', 'multiproducto', 'otro_producto']
    #            )
    #
    #df['lealtad'] = pd.Categorical(
    #            df.lealtad,
    #            ['volatil', 'ocasional', 'prefiere', 'leal']
    #            )
    #
    #df['rentab_categ'] = pd.Categorical(
    #            df.rentab_categ,
    #            ['rentable', 'no rentable', 'muy rentable', 'poco rentable']
    #            )
    
    df['predict_python'] = pd.Categorical(
                df.predict_python,
                ['cl-3', 'cl-1', 'cl-2']
                )
    
    df['seg_cliente_anual'] = pd.Categorical(
                df.seg_cliente_anual,
                ['missing', 'grandes', 'pymes']
                )
    
    df['emergente_anual'] = pd.Categorical(
                df.emergente_anual,
                ['missing', 'emergente', 'no_emergente', 'no_identificado']
                )
    
    df['capa_cliente_anual'] = pd.Categorical(
                df.capa_cliente_anual,
                ['missing', 'oro', 'bronce', 'plata']
                )
    
    df['min_cal_anual'] = pd.Categorical(
                df.min_cal_anual,
                ['missing', 'A', 'B', 'C', 'D', 'E', 'K']
                )
    
    df['min_cal_trim_ant'] = pd.Categorical(
                df.min_cal_trim_ant,
                ['missing', 'A', 'B', 'C', 'D', 'E', 'K']
                )
    
    df['max_cal_anual'] = pd.Categorical(
                df.max_cal_anual,
                ['missing', 'A', 'B', 'C', 'D', 'E', 'K']
                )
    
    df['max_cal_trim_ant'] = pd.Categorical(
                df.max_cal_trim_ant,
                ['missing', 'A', 'B', 'C', 'D', 'E', 'K']
                )

    # Separate in two models for 'activos' and 'pasivos' based on either:
    
    # 'x' represents the training set and 'y' the target variable
    
    vars_cat = [#'activo_pasivo','mono_multi','lealtad','rentab_categ',
            'predict_python','seg_cliente_anual',
            'emergente_anual','capa_cliente_anual','min_cal_anual','min_cal_trim_ant','max_cal_anual','max_cal_trim_ant']

    vars_activo = ['leasing_bin','cartera_ordinaria_bin','tarjeta_credito_bin',
               'creditos_tesoreria_bin','unidirecto_bin','factoring_bin']

    vars_pasivo = ['cuenta_corriente_bin','cuenta_ahorros_bin']
    
    vars_maybe = ['bienes_consumo_intermedio','caf','cartera_fomento','cartera_fomento_autos_&_bc','cartera_preoperativa','cdt',
                  'credito_empleados','credito_hipotecario','credito_rotativo','deudores_en_me','fomento_en_me','libranza','motos',
                  'prestamo_personal','rec_congelado_rem','recaudo_de_impuestos','sobregiro','vehiculos','bienes_consumo_intermedio_bin',
                  'caf_bin','cartera_fomento_bin','cartera_fomento_autos_&_bc_bin','cartera_preoperativa_bin','cdt_bin',
                  'credito_empleados_bin','credito_hipotecario_bin','credito_rotativo_bin','deudores_en_me_bin','fomento_en_me_bin',
                  'libranza_bin','motos_bin','prestamo_personal_bin','rec_congelado_rem_bin','recaudo_de_impuestos_bin','sobregiro_bin',
                  'vehiculos_bin']

    vars_num = [col for col in df.columns if (col not in vars_cat) and ('str_identificacion' not in col) and
                                         (col not in vars_activo) and (col not in vars_pasivo)]

    x_activo = df[[col for col in df.columns if ((col in vars_num) or (col in vars_cat)) and (col not in vars_maybe)]]
    x_pasivo = df[[col for col in df.columns if ((col in vars_num) or (col in vars_cat)) and (col not in vars_maybe)]]

    y_activo = df[vars_activo]
    y_pasivo = df[vars_pasivo]

    # Get dummies

    x_activo = pd.get_dummies(
            x_activo,
            columns=x_activo.columns[
                    np.where(x_activo.dtypes == 'category')[0]
                    ],
            drop_first=True
            )

    x_pasivo = pd.get_dummies(
            x_pasivo,
            columns=x_pasivo.columns[
                    np.where(x_pasivo.dtypes == 'category')[0]
                    ],
            drop_first=True
            )

    # Return df for both models and target variables

    return x_activo, x_pasivo, y_activo, y_pasivo

def get_file(file_path: str) -> pd.DataFrame:
    """
    Reads csv file in standardized format.

    :param file_path: str of file path
    :returns: df containing data
    """
    #print(select_cols)
    df = pd.read_csv(
            file_path,
            sep='|',
            #index_col=0,
            encoding='utf-8',
            na_values='',
            low_memory=False,
            #usecols = select_cols,
            decimal=',',
            error_bad_lines=False
            )

    return df

def get_file(file_path: str) -> pd.DataFrame:
    """
    Reads csv file in standardized format.

    :param file_path: str of file path
    :returns: df containing data
    """
    #print(select_cols)
    df = pd.read_csv(
            file_path,
            sep='|',
            #index_col=0,
            encoding='utf-8',
            na_values='',
            low_memory=False,
            #usecols = select_cols,
            decimal=',',
            error_bad_lines=False
            )

    return df