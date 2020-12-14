import pandas as pd
import pymysql
from sqlalchemy import create_engine

db_info = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'port': 3306,
    'database': 'text_classification'
}

engine = create_engine(
    'mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)d/%(database)s?charset=utf8' % db_info, encoding='utf-8')


def loadCsvIntoMysql(filepath):
    data = pd.read_csv(filepath, low_memory=False)

    # 可能字段名对不上，需要按照excel修改
    data = data[['统计', '新件名称整理', '故障描述(报修单)', '维修描述', '缺陷描述', '故障现象',
                 '故障原因', '故障排查过程', '故障系统', '故障件']]
    data = data.dropna(axis=0,subset = ['统计'])
    data['新件名称整理'] = data['新件名称整理'].astype('str').apply(lambda x: x.split(';')[0])
    data['新件名称整理'] = data['新件名称整理'].astype('str').apply(lambda x: x.split('；')[0])
    data.columns = ['fault_element', 'new_element', 'fault_desc', 'repair_desc', 'defect_desc', 'fault_phen',
                    'fault_reason', 'troubleshooting_process', 'fault_system', 'fault_parts']
    pd.io.sql.to_sql(data, 'quality_clear_data', con=engine, index=False, if_exists='append')


if __name__ == '__main__':
    loadCsvIntoMysql(r'./quality/2014年质量外反馈报表-发大数据.csv')
    loadCsvIntoMysql(r'./quality/2017年质量外反馈报表-发大数据.csv')
    loadCsvIntoMysql(r'./quality/2013年质量外反馈报表-发大数据.csv')
    loadCsvIntoMysql(r'./quality/2019年质量外反馈报表-发大数据.csv')
    loadCsvIntoMysql(r'./quality/2020年质量外反馈报表-发大数据.csv')

    pass
