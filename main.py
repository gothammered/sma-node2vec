import pandas as pd
import networkx as nx
import numpy as np

from tqdm import tqdm

"""
기본 설정 파트
"""
# 열 출력 제한 수 조정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 한글은 matplotlib 출력을 위해서 따로 설정이 필요하므로 font매니저 import
import matplotlib.font_manager

# 한글 출력을 위하여 font 설정
font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/HANDotum.ttf').get_name()
matplotlib.rc('font', family=font_name)

def tryNode2Vec(t_year):
    df = pd.read_csv('./data/transit_survey_SMA_{0}.csv'.format(t_year))
    adm_cd_list = df['ADM_CD_O'].unique().tolist()

    df_od_pivot = pd.pivot_table(df, index='ADM_CD_O', columns='ADM_CD_D', values='Unnamed: 0', aggfunc='count')

    G = nx.Graph()

    for adm_cd_o in tqdm(adm_cd_list):
        for adm_cd_d in adm_cd_list:
            w = df_od_pivot.at[adm_cd_o, adm_cd_d]
            if np.isnan(w):
                continue
            else:
                G.add_edge(adm_cd_o, adm_cd_d, weight=w)






for t_year in ['2006', '2010', '2016']:
    tryNode2Vec(t_year)