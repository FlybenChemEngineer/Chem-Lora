`bin_calculator.py`用于分箱操作。

```python
# 选择 case1, case2, case3 之一
case = 'case3'  # 可以换成 'case1' 或 'case2'
column_name = 'Yield' if case == 'case1' else 'L/B' if case == 'case2' else 'output'

# 进行分箱和映射操作
binned_data, bin_mapping = bin_data(case, column_name, num_bins=100)
mapped_data = map_binned_data(case)

# 输出结果
print(mapped_data[[column_name, 'bins', 'mean_value']].head())
```

通过更改case1, case2, case3对相应的数据进行分bin,mean_value是分箱后，与真实值的映射，这里取的是中值。

`csv_to_json.py`用于在执行分箱操作后生成训练的数据。

```python
main(case_num=3, input_type='train')
```

通过指定 case_num 和 train or test 生成相应案例的训练数据。

最后，SFT后的lora文件夹中找到`.jsonl`文件，放入`files`文件夹中。

运行`main_evaluate.py`。

```python
if __name__ == "__main__":
    # 直接设置 case 和 calculate_on 的值
    case = '1'  # '1' 或 '2' 或 '3'
    calculate_on = 'test'  # 或 'train'
    
    # 生成 JSONL 文件路径
    jsonl_file = f'files/test_case1_1_L231B_generated_predictions.jsonl'
    
    # 调用主函数
    main(case, jsonl_file, calculate_on)
```

修改这部分可计算相应案例的指标。