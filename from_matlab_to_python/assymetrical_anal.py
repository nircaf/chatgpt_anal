import numpy as np
import scipy.io
import pandas as pd
from scipy.stats import ttest_1samp

def read_mat(filename):
    # read .mat file
    mat = scipy.io.loadmat(filename)
    return mat

if __name__ == '__main__':
    # read .mat file
    mat = read_mat('epilepsy_controls_g_f.mat')
    # get restults mat lin age
    mat_lin = mat['result_mat_lin_age']
    # convert to df
    df_lin = pd.DataFrame(mat_lin).T
    # put mat['areas_names'] as index
    area_names = mat['areas_names'].copy()
    arr = []
    # convert to list
    for i in range(len(area_names)):
        # convert to list
        arr.append(area_names[i].tolist()[0].tolist()[0])
    df_lin.index = arr
    areas_left_right_index = 61
    # remove sides from areas
    areas_no_sides = [d[d.index(' ')+1:] for d in arr[4:int(2+len(area_names)/2)]]
    df_mean_sides = pd.DataFrame(index=areas_no_sides, columns=df_lin.columns)
    for i in range(4,int(2+len(area_names)/2)):
        # mean left and right
        df_mean_sides.loc[areas_no_sides[i-4]] = (df_lin.iloc[i,:] + df_lin.iloc[i+areas_left_right_index,:])/2
        # remove rows with all nan
    df_mean_sides = df_mean_sides.dropna(how='all')
    # fill nan with 0
    df_mean_sides = df_mean_sides.fillna(0)
    # create dict for p values with df_mean_sides.index as keys
    p_values = {}
    # perform t-test for each row
    for _, row in df_mean_sides.iterrows():
        # create a DataFrame with the rest of the rows
        other_rows = df_mean_sides[df_mean_sides.index != row.name]
        # flatten the rows into a single array
        other_values = other_rows.values.flatten()
        # perform t-test
        t_statistic, p_value = ttest_1samp(other_values, row.mean())
        assert np.mean(other_values) != np.nan
        p_values[row.name] = p_value

    # adjust p-values for multiple comparisons
    from statsmodels.stats.multitest import multipletests
    p_corrected = multipletests(list(p_values.values()), method='bonferroni')[1]
    p_values_corrected = {k:v for k,v in zip(p_values.keys(), p_corrected)}
    # count significant areas
    count = 0
    # print significant areas
    for key, value in p_values_corrected.items():
        if value < 0.05:
            print(f"Row {key} has a significant difference from the rest of the rows.")
            count += 1
    print(f"Number of significant areas: {count}")
    # get restults mat tofts age
    mat_tofts = mat['result_mat_tofts_age']
    print('done')
