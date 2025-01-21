import pandas as pd
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def bin_data(case, column_name, num_bins=100):
    # 根据 case 选择不同的文件路径
    file_path = f'data/{case}/{case}.csv'
    output_binned_file = f'data/{case}/{case}_binned_data.csv'
    output_mapping_file = f'data/{case}/{case}_bin_mapping.csv'
    
    # 自动检测文件编码
    encoding = detect_encoding(file_path)
    
    # 读取数据
    data = pd.read_csv(file_path, encoding=encoding)
    
    # 计算分箱
    data['bins'] = pd.qcut(data[column_name], q=num_bins, labels=False, duplicates='drop')
    
    # 创建包含所有 bin 的完整映射
    all_bins = pd.DataFrame({'bins': range(num_bins)})  # 创建完整的 bin 列表
    bin_mapping = data.groupby('bins')[column_name].mean().reset_index()
    bin_mapping.columns = ['bins', 'mean_value']
    
    # 确保 bin_mapping 包含所有 bin，即使没有数据的 bin 也存在，均值为 NaN
    complete_bin_mapping = all_bins.merge(bin_mapping, on='bins', how='left')
    
    # 填充空值：用前一个 bin 的均值填充空值
    complete_bin_mapping['mean_value'].fillna(method='ffill', inplace=True)

    # 保存映射文件和分箱数据
    complete_bin_mapping.to_csv(output_mapping_file, index=False, encoding='utf-8')
    data.to_csv(output_binned_file, index=False, encoding='utf-8')
    
    # 打印所有的 bin 分类对照表
    print("Bin Classification Mapping:")
    print(complete_bin_mapping)
    
    return data, complete_bin_mapping

def map_binned_data(case):
    file_path = f'data/{case}/{case}_binned_data.csv'
    mapping_file = f'data/{case}/{case}_bin_mapping.csv'
    
    # 自动检测文件编码
    encoding = detect_encoding(file_path)
    mapping_encoding = detect_encoding(mapping_file)
    
    # 读取数据
    data = pd.read_csv(file_path, encoding=encoding)
    bin_mapping = pd.read_csv(mapping_file, encoding=mapping_encoding)
    
    # 将映射文件中的平均值合并到原数据中
    data = data.merge(bin_mapping, on='bins', how='left')
    
    return data

# 示例用法
# 选择 case1, case2, case3 之一
case = 'case3'  # 可以换成 'case1' 或 'case2'
column_name = 'Yield' if case == 'case1' else 'L/B' if case == 'case2' else 'output'

# 进行分箱和映射操作
binned_data, bin_mapping = bin_data(case, column_name, num_bins=100)
mapped_data = map_binned_data(case)

# 输出结果
print(mapped_data[[column_name, 'bins', 'mean_value']].head())

