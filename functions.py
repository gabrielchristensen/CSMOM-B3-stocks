import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Limpa o dataframe dado
def clean_df(df):
    cleaned_df = df.copy()
    
    if ("Date" in df.columns):
        cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"])
        cleaned_df.set_index('Date', inplace = True)
    
    
    for i in cleaned_df.columns:
        cleaned_df[i] = pd.to_numeric(cleaned_df[i], errors='coerce')
    
    
    b3 = cleaned_df.pct_change()
    outliers = []

    for c in b3.columns:
        if b3[c].max() > 2:
            outliers.append(c)

    cleaned_df.drop(columns=outliers, inplace=True)
    
    
    return cleaned_df
       
def get_daily_return(df):  
    return df.pct_change().copy()

def get_monthly_return(ret_d):
    prices_m = ret_d.resample('BM').agg(lambda x: (x+1).prod() - 1)
    return prices_m

def get_momentum(ret_m, n=12, skip=1):
    return (ret_m.shift(skip)+1).rolling(n-skip).apply(np.prod, raw=True) - 1

def get_winners(mom, date, size=10):
    return mom.loc[date].nlargest(size)



def get_monthly_portfolio_return(formation, winners,  ret_m):
    # 1) descobrir o mês seguinte à formação
    iloc = ret_m.index.get_loc(formation)
    if isinstance(iloc, slice):
        iloc = iloc.start
    if iloc + 1 >= len(ret_m.index):
        return None
    next_month = ret_m.index[iloc + 1]

    # 2) validar que há ativos selecionados e que existem nas colunas de ret_m
    if len(winners) == 0:
        return None

    cols_w = [c for c in winners.index if c in ret_m.columns]

    if len(cols_w) == 0:
        return None

    # 3) calcular retorno igual-ponderado no próximo mês
    w_ret = ret_m.loc[next_month, cols_w].dropna().mean()
    if pd.isna(w_ret):
        return None

    return {
        "date": next_month,
        "winners_long": w_ret
    }

def backtest_momentum_longshort(df_prices,
                                top_n=10,
                                lookback=12,
                                skip=1,
                                min_universe=30):
    
    # 1) preparar dados de entrada do modelo
    ret_d = get_daily_return(df_prices)
    ret_m = get_monthly_return(ret_d)              # retorna mensal por produto dos diários no mês
    mom   = get_momentum(ret_m, n=lookback, skip=skip)  # momentum 12-1 típico

    results = []
    # 2) percorrer as datas de formação
    # começamos em mom.index[lookback:] para garantir janela suficiente
    for date in mom.index[lookback:]:
        # 2a) universo válido (sem NaN) e com tamanho mínimo
        valid = mom.loc[date].dropna()
        if len(valid) < max(min_universe, 2 * top_n):
            continue

        # 2b) formar carteiras (rank cross-sectional nessa data)
        winners = valid.nlargest(top_n)

        # 2c) medir retorno no mês seguinte (sem olhar para frente)
        res = get_monthly_portfolio_return(date, winners, ret_m)
        if res is not None:
            results.append(res)

    # 3) montar DataFrame de resultados
    if not results:
        return pd.DataFrame(columns=["winners_long"])

    bt = pd.DataFrame(results).set_index("date").sort_index()

    # 4) curvas acumuladas (base 1.0)
    bt["winners_cum"] = (1.0 + bt["winners_long"]).cumprod()
    return bt

def simulation(graph=True, save_csv=False):
    df = pd.read_csv("precos_b3_202010-2024_adjclose.csv")
    ibov_df_raw = pd.read_csv("ibov_2010_2024.csv")
    ibov_df = clean_df(ibov_df_raw)
    ibov_df = get_monthly_return(get_daily_return(ibov_df))
    ibov_df["Ibov_cReturn"] = (1.0 + ibov_df["Close"]).cumprod()
    ibov_df.rename(columns={'Close': 'Ibov_Return'}, inplace=True)
    prices = clean_df(df)
    bt = backtest_momentum_longshort(prices)
    bt = ibov_df.join(bt, how='inner')
    
    
    if graph:
        plt.figure(figsize=(12,6))
        plt.plot(bt.index, bt["winners_cum"], label="Estratégia", linewidth=2)
        plt.plot(bt.index, bt["Ibov_cReturn"], label="Ibov Return", linewidth=2)

        plt.title("Comparação CSMOM e Ibov", fontsize=14)
        plt.xlabel("Data")
        plt.ylabel("Índice acumulado (base = 1.0)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    if save_csv:
        bt.to_csv('ibov_x_momentum.csv', index=True)
    
    return bt


simulation()
