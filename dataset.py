# from dataset_day import *
from dataset_day_factor import *
# from dataset_min import *
# from dataset_multi import *
# from dataset_min_interday import *
# from dataset_barra import *
from torch.utils.data import DataLoader
import logging

# def get_min_day_data(data_config, df_min, df_day, data_type = ["train", "valid", "test"]):
#     df_day = df_day.dropna(axis=0, how='any')
#     y_label = ['$high', '$open', '$low', '$close', '$vwap']
#     df_day_y = df_day[y_label]
#     df_day_x = df_day.drop(columns = y_label)
    
#     train_set = Day_Min_Raw(data_frame_min=df_min, data_frame_day_x=df_day_x, data_frame_day_y=df_day_y, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T_day = data_config.T_day, T_min = data_config.T_min, H = data_config.label.H,
#                                 label_type=data_config.label.type, pct=data_config.pct_mode)
    
#     valid_set = Day_Min_Raw(data_frame_min=df_min, data_frame_day_x=df_day_x, data_frame_day_y=df_day_y, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T_day = data_config.T_day, T_min = data_config.T_min, H = data_config.label.H,
#                                 label_type=data_config.label.type, pct=data_config.pct_mode)
    
#     test_set = Day_Min_Raw(data_frame_min=df_min, data_frame_day_x=df_day_x, data_frame_day_y=df_day_y, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T_day = data_config.T_day, T_min = data_config.T_min, H = data_config.label.H,
#                                 label_type=data_config.label.type, pct=data_config.pct_mode)
#     output = []
#     if "train" in data_type:
#         output.append(train_set)
#     if "valid" in data_type:
#         output.append(valid_set)
#     if "test" in data_type:
#         output.append(test_set)
    
#     return output
    
# def get_min_raw_data(data_config, df_min, df_day, data_type = ["train", "valid", "test"]):
#     train_set = Data_Min_Raw(data_frame_min=df_min, data_frame_day=df_day, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, pct=data_config.pct_mode)
#     valid_set = Data_Min_Raw(data_frame_min=df_min, data_frame_day=df_day, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, pct=data_config.pct_mode)
#     test_set = Data_Min_Raw(data_frame_min=df_min, data_frame_day=df_day, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, pct=data_config.pct_mode)
#     output = []
#     if "train" in data_type:
#         output.append(train_set)
#     if "valid" in data_type:
#         output.append(valid_set)
#     if "test" in data_type:
#         output.append(test_set)
    
#     return output

def get_day_factor_data(data_config, df_data, data_type = ["train", "valid", "test"]):
    
    df_data = df_data.dropna(axis=0, how='any')
    y_label = ['close']
    df_y = df_data[y_label]
    df_x = df_data.drop(columns = y_label)
    output = []
    training_set = Day_Factor_new(data_frame_x=df_x, data_frame_y = df_y, fields=data_config.factors,
                                train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
                                valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
                                test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
                                mode = "train", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, label_weight = data_config.label.weight, 
                            pct=data_config.pct_mode)
        
    valid_set = Day_Factor_new(data_frame_x=df_x, data_frame_y = df_y, fields=data_config.factors,
                            train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
                            valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
                            test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
                            mode = "valid", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, label_weight = data_config.label.weight, 
                            pct=data_config.pct_mode)
    
    test_set = Day_Factor_new(data_frame_x=df_x, data_frame_y = df_y, fields=data_config.factors,
                            train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
                            valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
                            test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
                            mode = "test", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, label_weight = data_config.label.weight, 
                            pct=data_config.pct_mode)
        

    if "train" in data_type:
        output.append(training_set)
    if "valid" in data_type:
        output.append(valid_set)
    if "test" in data_type:
        output.append(test_set)
    
    return output

# def get_day_data(data_config, df_data, data_type = ["train", "valid", "test"]):
    
#     if data_config.load_type == 'old':
#         df2 = df_data.reset_index()
#         time_points = df2["datetime"].unique()
#         dict_data = {f:df_data[f].unstack() for f in data_config.factors} # 每一天 每一支股票的属性
        
#         print("loading train/valid/test set ,,, ,, ")
        
#         output = []
#         training_set = Day_Data(dict_data=dict_data, timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.T, H = data_config.label.H)
        
#         valid_set = Day_Data(dict_data=dict_data, timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.T, H = data_config.label.H)
        
#         test_set = Day_Data(dict_data=dict_data, timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.T, H = data_config.label.H)
        
#     elif data_config.load_type == 'new':
#         output = []
        
#         training_set = Day_Data_new(data_frame=df_data, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, pct=data_config.pct_mode)
        
#         valid_set = Day_Data_new(data_frame=df_data, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, pct=data_config.pct_mode)
        
#         test_set = Day_Data_new(data_frame=df_data, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, pct=data_config.pct_mode)
        

#     elif data_config.load_type == 'new_cache':
#         output = []
#         res_x, y = None, None
        
#         # df_data = df_data.loc["2011-01-01":"2012-01-01"]
#         df_able_instrument=get_able_instrument(df_data.close.unstack().sort_index(), data_config.T + data_config.label.H + 2).shift(- (data_config.label.H + 1))
#         df_able_instrument=df_able_instrument.replace(False,np.nan)
#         all_dates = df_able_instrument.index
        
#         res_l=[]
#         start_attr, end_attr = data_config.label.type.split('2')
#         tmp = df_data[data_config.factors].groupby('instrument')
#         base=tmp.shift(data_config.T)
#         y=tmp[end_attr].shift(-(data_config.label.H+1))/tmp[start_attr].shift(-1)-1
#         y = y.groupby('datetime').rank(pct=True)
#         able_index=df_able_instrument.stack().sort_index().index
#         for i in tqdm(range(0, data_config.T)):
#             ii=tmp.shift(i)/base
#             ii=ii.loc[able_index]
#             res_l.append(ii)
        
#         print("concating ... ")
        
#         res_all=pd.concat(res_l)
#         res_x = res_all
#         y = y[able_index].sort_index()
        
        
#         training_set = Day_Data_new_cache(data_frame=df_data, fields=data_config.factors, res_x=res_x, res_y=y, all_dates=all_dates,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, df_able_instrument=df_able_instrument)
        
#         valid_set = Day_Data_new_cache(data_frame=df_data, fields=data_config.factors, res_x=res_x, res_y=y, all_dates=all_dates,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, df_able_instrument=df_able_instrument)
        
#         test_set = Day_Data_new_cache(data_frame=df_data, fields=data_config.factors, res_x=res_x, res_y=y, all_dates=all_dates,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, df_able_instrument=df_able_instrument)
    
#     elif data_config.load_type == 'cache':
#         output = []
#         cache_data = pd.read_pickle(data_config.cache_data)
#         cache_y = pd.read_pickle(data_config.cache_y)
#         time_points = cache_y.reset_index()['datetime'].unique()
#         training_set = Day_Data_Cache(cache_data=cache_data,cache_y=cache_y ,timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.T, H = data_config.label.H, fast=data_config.fast_mode)
        
#         valid_set = Day_Data_Cache(cache_data=cache_data,cache_y=cache_y , timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.T, H = data_config.label.H)
        
#         test_set = Day_Data_Cache(cache_data=cache_data,cache_y=cache_y , timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.T, H = data_config.label.H)
#     elif data_config.load_type == 'mult_freq':
#         output =[]
#         cache_data_day = pd.read_pickle(data_config.cache_data_day)
#         cache_y = pd.read_pickle(data_config.cache_y)
#         cache_data_week = pd.read_pickle(data_config.cache_data_week)
#         cache_data_month = pd.read_pickle(data_config.cache_data_month)
#         time_points = cache_y.reset_index()['datetime'].unique()
#         training_set_day = Day_Data_Cache(cache_data=cache_data_day,cache_y=cache_y ,timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.T, H = data_config.label.H, fast=data_config.fast_mode)
#         valid_set_day = Day_Data_Cache(cache_data=cache_data_day,cache_y=cache_y , timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.T, H = data_config.label.H)
#         test_set_day = Day_Data_Cache(cache_data=cache_data_day,cache_y=cache_y , timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.T, H = data_config.label.H)
#         training_set_week = Day_Data_Cache(cache_data=cache_data_week,cache_y=cache_y ,timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.week_T, H = data_config.label.H, fast=data_config.fast_mode)
#         valid_set_week = Day_Data_Cache(cache_data=cache_data_week,cache_y=cache_y , timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.week_T, H = data_config.label.H)
#         test_set_week = Day_Data_Cache(cache_data=cache_data_week,cache_y=cache_y , timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.week_T, H = data_config.label.H)
#         training_set_month = Day_Data_Cache(cache_data=cache_data_month,cache_y=cache_y ,timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.month_T, H = data_config.label.H, fast=data_config.fast_mode)
#         valid_set_month = Day_Data_Cache(cache_data=cache_data_month,cache_y=cache_y , timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.month_T, H = data_config.label.H)
#         test_set_month = Day_Data_Cache(cache_data=cache_data_month,cache_y=cache_y , timepoints=time_points, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.month_T, H = data_config.label.H)
#         if "train" in data_type:
#             output.append(training_set_day)
#             output.append(training_set_week)
#             output.append(training_set_month)
#         if "valid" in data_type:
#             output.append(valid_set_day)
#             output.append(valid_set_week)
#             output.append(valid_set_month)
#         if "test" in data_type:
#             output.append(test_set_day)
#             output.append(test_set_week)
#             output.append(test_set_month)
#             return output 
        

        

#     if "train" in data_type:
#         output.append(training_set)
#     if "valid" in data_type:
#         output.append(valid_set)
#     if "test" in data_type:
#         output.append(test_set)
    
#     return output

# def get_inter_day_data(data_config, df_min, data_type = ["train", "valid", "test"]):
#     train_set = Data_inter_Min_Raw(data_frame_min=df_min, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, pct=data_config.pct_mode)
#     valid_set = Data_inter_Min_Raw(data_frame_min=df_min, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, pct=data_config.pct_mode)
#     test_set = Data_inter_Min_Raw(data_frame_min=df_min, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.T, H = data_config.label.H, label_type=data_config.label.type, pct=data_config.pct_mode)
#     output = []
#     if "train" in data_type:
#         output.append(train_set)
#     if "valid" in data_type:
#         output.append(valid_set)
#     if "test" in data_type:
#         output.append(test_set)
    
#     return output


# def get_day_barra_data(data_config, df_barra, df_day, data_type = ["train", "valid", "test"]):

#     y_label = ['$high', '$open', '$low', '$close', '$vwap']
#     df_day_y = df_day[y_label]
#     df_day_x = df_day.drop(columns = y_label)
    
    
#     train_set = Day_Barra_Raw(data_frame_barra=df_barra, data_frame_day_x=df_day_x, data_frame_day_y=df_day_y, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "train", T = data_config.T, H = data_config.label.H,label_type=data_config.label.type, label_weight = data_config.label.weight,
#                                 pct=data_config.pct_mode)
#     valid_set = Day_Barra_Raw(data_frame_barra=df_barra, data_frame_day_x=df_day_x, data_frame_day_y=df_day_y, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "valid", T = data_config.T, H = data_config.label.H,label_type=data_config.label.type, label_weight = data_config.label.weight,
#                                 pct=data_config.pct_mode)

#     test_set = Day_Barra_Raw(data_frame_barra=df_barra, data_frame_day_x=df_day_x, data_frame_day_y=df_day_y, fields=data_config.factors,
#                                 train_start=data_config.train_test_split.train_start, train_end=data_config.train_test_split.train_end,
#                                 valid_start=data_config.train_test_split.valid_start, valid_end=data_config.train_test_split.valid_end,
#                                 test_start=data_config.train_test_split.test_start, test_end=data_config.train_test_split.test_end,
#                                 mode = "test", T = data_config.T, H = data_config.label.H,label_type=data_config.label.type, label_weight = data_config.label.weight,
#                                 pct=data_config.pct_mode)

#     output = []
#     if "train" in data_type:
#         output.append(train_set)
#     if "valid" in data_type:
#         output.append(valid_set)
#     if "test" in data_type:
#         output.append(test_set)
    
#     return output



def get_loader(data_config, mode = 'train'):
    if data_config.freq == 'day':
        if data_config.data_type == 'raw': # reading raw data
            if data_config.load_type in ['old', 'new', 'new_cache']:
                df = load_data(data_config)
                train_data, valid_data, test_data = get_day_data(data_config, df)
            
            elif data_config.type == 'cache':
                logging.info("Reading cache ... this may be really slow ...")
                
                train_data, valid_data, test_data = get_day_data(data_config,df_data=None)
            elif data_config.data_type == "mult_freq":
                logging.info("Reading multi freq data ... this may be really slow ...")
                (train_data_day, train_data_week,train_data_month,valid_data_day,valid_data_week,valid_data_month,
                test_data_day,test_data_week,test_data_month) = get_day_data(data_config,df_data=None)
                train_data_loader1 = DataLoader(train_data_day, batch_size=1, shuffle=True, num_workers=1)
                train_data_loader2 = DataLoader(train_data_week, batch_size=1, shuffle=True, num_workers=1)
                train_data_loader3 = DataLoader(train_data_month, batch_size=1, shuffle=True, num_workers=1)
                valid_data_loader1 = DataLoader(valid_data_day, batch_size=1, shuffle=False, num_workers=1)
                valid_data_loader2 = DataLoader(valid_data_week, batch_size=1, shuffle=False, num_workers=1)
                valid_data_loader3 = DataLoader(valid_data_month, batch_size=1, shuffle=False, num_workers=1)
                test_data_loader1 = DataLoader(test_data_day, batch_size=1, shuffle=False, num_workers=1)
                test_data_loader2 = DataLoader(test_data_week, batch_size=1, shuffle=False, num_workers=1)
                test_data_loader3 = DataLoader(test_data_month, batch_size=1, shuffle=False, num_workers=1)
                return train_data_loader1, train_data_loader2, train_data_loader3, valid_data_loader1, valid_data_loader2, valid_data_loader3, test_data_loader1, test_data_loader2, test_data_loader3


        elif data_config.data_type == 'factor': # data type is factors
            if data_config.load_type in ['new']:
                df = pd.read_pickle(data_config.factor_data_path)
                
                #df = load_factor_data(df,data_config)
                print(df.shape)
                train_data, valid_data, test_data = get_day_factor_data(data_config, df)
            elif data_config.load_type in ['barra']:
                df_barra, df_day = load_barra_day_data(data_config)

                train_data, valid_data, test_data = get_day_barra_data(data_config, df_barra,df_day)
        else:
            raise ValueError(f"Expected data type: factor or raw, but received {data_config.data_type}")
    
    elif data_config.freq == 'min':
        df_min, df_day = load_min_data(data_config, mode=mode)
        train_data, valid_data, test_data = get_min_raw_data(data_config=data_config, df_min=df_min, df_day=df_day)
    elif data_config.freq in ['min_day', 'day_min']:
        df_min, df_day = load_min_day_data(data_config, mode=mode)
        train_data, valid_data, test_data = get_min_day_data(data_config=data_config, df_min=df_min, df_day=df_day)
    
    elif data_config.freq == 'inter_day':
        df_min = load_min_data(data_config, mode=mode)
        train_data, valid_data, test_data = get_inter_day_data(data_config=data_config, df_min=df_min)
    
    else:
        raise ValueError("Not implemented yet!!!")
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=16)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=16)
    
    return train_loader, valid_loader, test_loader

    
if __name__ == "__main__":
    # import pandas as pd

    # data = pd.read_pickle("63_day_data_after_filtering.pkl")
    # print(data.loc['2023-12-05'])
    config = load_config("configs/data_config_day_factor.yaml")
    
    df = pd.read_pickle(config.factor_data_path)

    # df = load_data(config)
    valid_data, test_data = get_day_factor_data(config, df, ['test', 'valid'])
    from torch.utils.data import DataLoader
    loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=16)
    # print(test_data[349])
    for (features, targets, today, able_instrument) in tqdm(loader):
        pass