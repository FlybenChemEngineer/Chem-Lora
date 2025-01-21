import pandas as pd
from rdkit import Chem
import json

def load_functional_groups_from_csv(csv_file):
    """Load functional group information from a CSV file, returning a dictionary of group names and SMARTS patterns."""
    fg_df = pd.read_csv(csv_file, header=None, names=['SMARTS', 'GroupName'])
    functional_groups = {row['SMARTS']: row['GroupName'] for _, row in fg_df.iterrows()}
    return functional_groups

def find_functional_groups(smiles: str, functional_groups: dict):
    """Extract functional group information from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    detected_groups = {}
    
    for group_smarts, group_name in functional_groups.items():
        patt = Chem.MolFromSmarts(group_smarts)
        if mol.HasSubstructMatch(patt):
            matches = mol.GetSubstructMatches(patt)
            detected_groups[group_name] = len(matches)
    
    return detected_groups

def generate_natural_language_description(smiles: str, detected_groups):
    """Generate a natural language description based on detected functional groups."""
    if not detected_groups:
        return f"{smiles} does not contain any functional groups."
    
    descriptions = []
    for group, count in detected_groups.items():
        if count == 1:
            descriptions.append(f"1 {group}")
        else:
            descriptions.append(f"{count} {group}")
    
    return ", ".join(descriptions) + "."

def detect_functional_groups_and_merge(input_csv, smiles_columns, functional_groups_csv, output_csv):
    """Process SMILES columns, detect functional groups, merge results into the original CSV file, and save."""
    ligands_df = pd.read_csv(input_csv, encoding='ISO-8859-1', on_bad_lines='skip')
    functional_groups = load_functional_groups_from_csv(functional_groups_csv)
    overall_descriptions = []

    for i, row in ligands_df.iterrows():
        combined_description = []
        for smiles_column in smiles_columns:
            smiles = row[smiles_column]
            try:
                detected_groups = find_functional_groups(smiles, functional_groups)
                natural_language_desc = generate_natural_language_description(smiles, detected_groups)
                combined_description.append(f"{smiles_column}, its SMILES is {smiles}, contains {natural_language_desc}")
            except ValueError as e:
                print(f"Error: {e}")
                combined_description.append(f"{smiles_column}, its SMILES is {smiles}, contains Invalid SMILES")

        overall_descriptions.append(", ".join(combined_description))

    # Add the combined description to the original dataframe
    ligands_df['Description'] = overall_descriptions

    # Save the merged result back into the original CSV file
    ligands_df.to_csv(output_csv, index=False)

def convert_csv_to_json(input_csv, output_json, reaction_type, description_template, output_column='bins'):
    """
    将CSV文件转换为目标JSON格式，并保存为指定文件。
    
    Parameters:
    - input_csv: 输入的CSV文件路径
    - output_json: 输出的JSON文件路径
    - reaction_type: 反应类型描述（用于 instruction 说明）
    - description_template: 描述模板，包含对CSV每列的解释，格式化字符串中引用的列名应与CSV列标题匹配
    - output_column: 指定输出结果的列名，默认为 'bins'
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 转换为目标格式
    data = []
    for _, row in df.iterrows():
        # 格式化描述信息
        description = description_template.format(**row)
        
        # 创建目标格式的字典
        entry = {
            "instruction": f"This is a {reaction_type} reaction. Description: {description}, what is the yield?",  
            "input": "",
            "output": f"{row[output_column]}",  # 使用CSV中的'bin'列或指定列
            "history": []
        }
        data.append(entry)

    # 保存为JSON文件
    with open(output_json, 'w', encoding='utf-8') as dataset_file:
        json.dump(data, dataset_file, ensure_ascii=False, indent=4)

def main(case_num, input_type):
    """Main function to process case data based on the case number and input type (train or test)."""
    
    # Define input and output filenames based on case number and type (train or test)
    input_csv = f'data/case{case_num}/case{case_num}_{input_type}_set.csv'
    output_csv = f'log/case{case_num}/case{case_num}_{input_type}_set_with_functional_groups.csv'
    functional_groups_csv = 'data/functional_groups_smiles_codes.csv'
    
    # Define the columns for each case
    if case_num == 1:
        smiles_columns = ['Ligand', 'Additive', 'Base', 'Aryl halide']
        reaction_type = 'B-H coupling'
        description_template = "{Description}"  # 仅使用Description列内容
    elif case_num == 2:
        smiles_columns = ['olefin', 'ligand', 'solvent']
        reaction_type = 'terminal olefin hydroformylation'
        description_template = ("The formed transition metal complex is {Ligand-M}, temperature is {T(K)} K, "
                                "the pressure is {p(bar)} bar, the CO/H2 is {CO/H2}, "
                                "the reaction time is {t(h)} h, the L/M is {L/M}, "
                                "the S/M is {S/M}, and the molar concentration of the alkene is {S/M}.")
    elif case_num == 3:
        smiles_columns = ['Catalyst', 'Imine', 'Thiol']
        reaction_type = 'chiral phosphoric acid-catalyzed'
        description_template = "{Description}"
    else:
        raise ValueError(f"Invalid case number: {case_num}")
    
    # Detect functional groups and merge into the CSV file
    detect_functional_groups_and_merge(input_csv, smiles_columns, functional_groups_csv, output_csv)
    
    # Convert the CSV to JSON
    json_output = f'log/case{case_num}/{input_type}_case{case_num}.json'
    convert_csv_to_json(output_csv, json_output, reaction_type, description_template)

# Example usage:
# main(case_num=1, input_type='train')
# main(case_num=2, input_type='train')
main(case_num=3, input_type='train')
