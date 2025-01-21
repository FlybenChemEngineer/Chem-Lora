import json
import csv
import re
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error,root_mean_squared_error

def jsonl_to_csv(input_file, output_file):
    def process_predict_value(predict):
        numbers = re.findall(r'\d+', predict)
        if len(numbers) >= 2:
            return numbers[0][:1] + numbers[1][:1]
        elif numbers:
            return numbers[0][:2]
        return ''

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['label', 'predict'])
        
        for line in infile:
            data = json.loads(line)
            label = data.get('label', '')
            predict = data.get('predict', '')
            processed_predict = process_predict_value(predict)
            csv_writer.writerow([label, processed_predict])

    print("CSV文件生成成功！")

def map_predict_to_mean_values(predict_file, bin_mapping_file, output_file):
    case_output = pd.read_csv(predict_file)
    predict_values = case_output['predict'].astype(str)
    bin_mapping = pd.read_csv(bin_mapping_file)
    bin_mapping['bins'] = bin_mapping['bins'].astype(str)

    mapped_values = predict_values.map(bin_mapping.set_index('bins')['mean_value'])
    unmatched_indices = predict_values[~predict_values.isin(bin_mapping['bins'])].index

    if not unmatched_indices.empty:
        print("未匹配的值的行号:", unmatched_indices.tolist())

    case_output['mean_values'] = mapped_values
    case_output.to_csv(output_file, index=False)
    print(f"映射完成，已保存为 {output_file}")

def evaluate_model(case_file, output_file, calculate_on='train'):
    case_data = pd.read_csv(case_file)
    
    # 根据 case 选择实际值列
    if 'Yield' in case_data.columns:
        actual_values = case_data['Yield']
    elif 'L/B' in case_data.columns:
        actual_values = case_data['L/B']
    elif 'output' in case_data.columns:
        actual_values = case_data['output']
    else:
        raise ValueError("未找到适用的实际值列。")

    case_output = pd.read_csv(output_file)
    mean_values = case_output['mean_values']

    nan_rows = mean_values[mean_values.isnull()].index
    print("NaN 行号:", nan_rows.tolist())

    r2 = r2_score(actual_values, mean_values)
    mae = mean_absolute_error(actual_values, mean_values)
    rmse = root_mean_squared_error(actual_values, mean_values)

    print(f"{'训练集' if calculate_on == 'train' else '测试集'}结果:")
    print(f"R²: {r2}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

def main(case, jsonl_file, calculate_on):
    # Step 1: JSONL to CSV
    csv_file = f'log/output_case{case}.csv'
    jsonl_to_csv(jsonl_file, csv_file)

    # Step 2: Map predict to mean values
    bin_mapping_file = f'data/case{case}/case{case}_bin_mapping.csv'
    output_with_mean_values = f'log/output_with_mean_values_case{case}.csv'
    map_predict_to_mean_values(csv_file, bin_mapping_file, output_with_mean_values)

    # Step 3: Evaluate the model
    test_set_file = f'data/case{case}/case{case}_test_set.csv'
    evaluate_model(test_set_file, output_with_mean_values, calculate_on)

if __name__ == "__main__":
    # 直接设置 case 和 calculate_on 的值
    case = '1'  # '1' 或 '2' 或 '3'
    calculate_on = 'test'  # 或 'train'
    
    # 生成 JSONL 文件路径
    jsonl_file = f'files/test_case1_1_L231B_generated_predictions.jsonl'
    
    # 调用主函数
    main(case, jsonl_file, calculate_on)

