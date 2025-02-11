import torch
from torch.utils.data import DataLoader
from dataset import *
from utils import *
from tqdm import tqdm
import argparse


def parse_arguments():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Parse command line arguments.')

    # 添加 config_name 参数，指定类型为字符串，没有提供默认值
    parser.add_argument('--test_config', type=str, default='./configs/test.yaml',
                        help='Name of the configuration file.')
    parser.add_argument('--data_config', type=str, default='./configs/data_config_new.yaml',
                        help='Name of the configuration file.')
    parser.add_argument('--mode', type=str, default='mean',
                        help='save mean or total factors')
    parser.add_argument('--save_file', type=str, required=True,
                        help='target save file.')
    parser.add_argument('--use_zscore', default=False, action='store_true',
                        help='whether use zscore when ensembling')
    
    # 解析命令行参数
    args = parser.parse_args()

    return args

class ModelList:
    def __init__(self) -> None:
        self.models = []
        
    def append(self, model):
        self.models.append(model)
        
def zscore_normalize(output_tensor):
    mean = torch.mean(output_tensor, dim=0, keepdim=True)
    std = torch.std(output_tensor, dim=0, keepdim=True)
    normalized_output = (output_tensor - mean) / (std + 1e-8)  # 添加一个小的常数以防止除零错误
    return normalized_output

def save_mean_factors(model_list, test_loader, model_key, use_zscore):
    # 初始化一个空的DataFrame，具有多层索引和指定的列（即模型名称）
    df = pd.DataFrame(columns=model_key, 
                      index=pd.MultiIndex.from_arrays([[], []], names=['datetime', 'instrument']))

    # 确保模型处于评估模式
    for model in model_list:
        model.eval()

    with torch.no_grad():
        for batch_idx, (features, targets, datetimes, instruments) in enumerate(tqdm(test_loader)):
            if torch.cuda.is_available():
                features = features.float().squeeze(0).cuda()

            repeated_datetimes = np.repeat(datetimes, len(instruments))
            # 为当前批次创建一个空的DataFrame来存储因子值
            current_batch_df = pd.DataFrame(index=pd.MultiIndex.from_arrays([repeated_datetimes, instruments], 
                                                                            names=['datetime', 'instrument']))

            for model_index, model in enumerate(model_list):
                outputs = model(features)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                if use_zscore:
                    normed_feature = zscore_normalize(outputs)
                    factors = (torch.mean(normed_feature, dim=1).cpu().tolist())
                else:
                    factors = torch.mean(outputs, dim=1).cpu().tolist()

                # 将因子值添加到当前批次的DataFrame中
                current_batch_df[model_key[model_index]] = factors

            # 将当前批次的结果合并到总的DataFrame中
            df = pd.concat([df, current_batch_df])
            # df['instrument'] = df['instrument'].apply(lambda x: x[2:-3])
            # df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def save_single_factors(model_list, test_loader, model_key, use_zscore):
    # 初始化一个空的DataFrame，具有多层索引和指定的列（即模型名称）
    df = pd.DataFrame(columns=model_key, 
                      index=pd.MultiIndex.from_arrays([[], []], names=['datetime', 'instrument']))

    # 确保模型处于评估模式
    for model in model_list:
        model.eval()

    with torch.no_grad():
        for batch_idx, (features, targets, datetimes, instruments) in enumerate(tqdm(test_loader)):
            if torch.cuda.is_available():
                features = features.float().squeeze(0).cuda()

            repeated_datetimes = np.repeat(datetimes, len(instruments))
            
            # 为当前批次创建一个空的DataFrame来存储因子值
            # 注意：这里需要为每个因子分配一个列名
            factor_columns = [f'factor_{i}' for i in range(64)]
            current_batch_df = pd.DataFrame(index=pd.MultiIndex.from_arrays([repeated_datetimes, instruments], 
                                                                            names=['datetime', 'instrument']),
                                            columns=factor_columns)

            for model_index, model in enumerate(model_list):
                outputs = model(features)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                if use_zscore:
                    outputs = zscore_normalize(outputs)

                # 保存所有因子而不是平均值
                # 将因子数据转换为列表格式，并存储到DataFrame中
                for i in range(outputs.shape[0]):
                    current_batch_df.loc[(repeated_datetimes[i], instruments[i]), factor_columns] = outputs[i].cpu().numpy()

            # 将当前批次的结果合并到总的DataFrame中
            df = pd.concat([df, current_batch_df])
            # df['instrument'] = df['instrument'].apply(lambda x: x[2:-3])
            # df['datetime'] = pd.to_datetime(df['datetime'])
    return df


if __name__ == '__main__':
    args = parse_arguments()
    test_config = load_config(args.test_config)
    data_config = load_config(args.data_config)
    
    train_loader, valid_loader, test_loader = get_loader(data_config, mode='valid')

    model_list = ModelList()
    
    for model_key, values in test_config.__dict__.items():
        model_list.append(get_model(values, False)[1].cuda())
    
    model_keys = list(test_config.__dict__.keys())
    df = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], []], names=['datetime', 'instrument']))
    
    if args.mode == 'single':
        df_valid = save_single_factors(model_list.models, valid_loader, model_keys, args.use_zscore)
        df_test = save_single_factors(model_list.models, test_loader, model_keys, args.use_zscore)
    elif args.mode == 'mean':
        df_valid = save_mean_factors(model_list.models, valid_loader, model_keys, args.use_zscore)
        df_test = save_mean_factors(model_list.models, test_loader, model_keys, args.use_zscore)
    else:
        raise ValueError("No such save mode!!!")
    df = pd.concat([df_valid, df_test])
    df['instrument'] = df['instrument'].apply(lambda x: x[2:-3])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.to_csv(f"factor_repo/{args.save_file}")
    # for i, (model_key, values) in enumerate(test_config.__dict__.items()):
    #     model_list.append(get_model(values, False).cuda())  
    #     save_factors(model_list.models[i], valid_loader, test_config.__dict__[model_key])
        # save_factors(model_list.models[i], valid_loader, test_config.__dict__[model_key])
        
    
    