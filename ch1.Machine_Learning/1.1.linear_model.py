
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import os

import sklearn.neighbors

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

if __name__ == "__main__":
    datapath = os.path.join("../", "datasets", "lifesat", "")

    # 데이터 적재
    oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
    gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

    # 데이터 준비
    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    x = np.c_[country_stats["GDP per capita"]]
    y = np.c_[country_stats["Life satisfaction"]]

    # 데이터 시각화
    country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
    plt.show()

    # 모델 선택
    #model = sklearn.linear_model.LinearRegression()     # 선형 회귀 모델
    model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)    # KNN 회귀 모델

    # 모델 훈련
    model.fit(x, y)

    # 키프로스에 대한 예측
    x_new = [[22587]]   # 키프로스 국가 1인 당 GDP
    print(model.predict(x_new))