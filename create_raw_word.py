import pandas as pd

if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv(r'data/quality/quality_feedback_fault_use38_all.csv', sep=',')
    df = df[['fault_element', 'new_element', 'fault_desc', 'defect_desc', 'fault_phen', 'fault_reason']]

    df['static'] = df['fault_element']
    df['sentence'] = df['fault_desc'] +","+ df['fault_phen'] +","+ df['fault_reason']
    df['sentence'] = df['sentence'].astype(str)
    df = df[['static', 'sentence']]

    df.to_csv(r'data/故障描述与标签.csv')