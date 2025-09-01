import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


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
    values = []

    for c in b3.columns:
        if b3[c].max() > 2:
            outliers.append(c)
            values.append(b3[c].max())
    print(outliers)
    print(values)
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

def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = series / roll_max - 1
    return drawdown.min()

def calc_metrics_year(bt, rf=0.0):

    results = []

    # agrupar por ano
    for year, data in bt.groupby(bt.index.year):
        if len(data) < 3:  # ignora anos incompletos
            continue

        ret = data["winners_long"].dropna()
        ibov_ret = data["Ibov_Return"].dropna()

        if len(ret) == 0:
            continue

        # CAGR (para aquele ano: compounding mensal)
        cagr = (1 + ret).prod() - 1

        # volatilidade anualizada
        vol = ret.std() * np.sqrt(12)

        # Sharpe ratio
        sharpe = (ret.mean() * 12 - rf) / vol if vol > 0 else np.nan

        # Sortino ratio
        downside = ret[ret < 0].std() * np.sqrt(12)
        sortino = (ret.mean() * 12 - rf) / downside if downside > 0 else np.nan

        # Max Drawdown (naquele ano)
        cum = (1 + ret).cumprod()
        mdd = max_drawdown(cum)

        # Calmar ratio
        calmar = cagr / abs(mdd) if mdd != 0 else np.nan

        # Hit ratio
        hit_ratio = (ret > 0).mean()

        # Beta e alpha (regressão contra Ibov)
        if len(ibov_ret) > 0 and len(ret) == len(ibov_ret):
            X = ibov_ret.values.reshape(-1, 1)
            y = ret.values
            reg = LinearRegression().fit(X, y)
            beta = reg.coef_[0]
            alpha = reg.intercept_
        else:
            beta, alpha = np.nan, np.nan

        # Correlação
        corr = ret.corr(ibov_ret) if len(ibov_ret) > 0 else np.nan

        # Information Ratio
        excess = ret - ibov_ret
        te = excess.std() * np.sqrt(12)
        ir = (excess.mean() * 12) / te if te > 0 else np.nan

        results.append({
            "Ano": year,
            "CAGR": cagr,
            "Volatilidade": vol,
            "Max Drawdown": mdd,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "HitRatio": hit_ratio,
            "Beta": beta,
            "Alpha": alpha,
            "Corr": corr,
            "InfoRatio": ir
        })
        
        

    return pd.DataFrame(results)


def calc_metrics_total(bt, rf=0.0):
    """
    Calcula métricas de performance totais para a estratégia e o Ibov.
    bt deve conter colunas:
        - winners_long   (retornos mensais da estratégia)
        - winners_cum    (curva acumulada da estratégia)
        - Ibov_Return    (retornos mensais do Ibov)
        - Ibov_cReturn   (curva acumulada do Ibov)
    rf = taxa livre de risco (mensalizado ou 0.0)
    """

    results = []


    for label, ret_col, cum_col in [
        ("Estrategia", "winners_long", "winners_cum"),
        ("Ibov", "Ibov_Return", "Ibov_cReturn")
    ]:
        ret = bt[ret_col].dropna()
        cum = bt[cum_col].dropna()

        if len(ret) == 0:
            continue

        # CAGR
        n_years = (cum.index[-1] - cum.index[0]).days / 365.25
        cagr = (cum.iloc[-1])**(1/n_years) - 1

        # Volatilidade anualizada
        vol = ret.std() * np.sqrt(12)

        # Sharpe Ratio
        sharpe = (ret.mean()*12 - rf) / vol if vol > 0 else np.nan

        # Sortino Ratio
        downside = ret[ret < 0].std() * np.sqrt(12)
        sortino = (ret.mean()*12 - rf) / downside if downside > 0 else np.nan

        # Max Drawdown
        mdd = max_drawdown(cum)

        # Calmar Ratio
        calmar = cagr / abs(mdd) if mdd != 0 else np.nan

        # Hit Ratio
        hit_ratio = (ret > 0).mean()

        results.append({
            "CAGR": cagr,
            "Volatilidade": vol,
            "Max Drawdown": mdd,
            "Índice de Sharpe": sharpe,
            "Índice de Sortino": sortino,
            "Índice de Calmar": calmar,
            "HitRatio": hit_ratio
        })

    return pd.DataFrame(results)


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

def graph_bt(bt, ibov):

    plt.figure(figsize=(12,6))
    plt.plot(bt.index, bt["winners_cum"], label="Estratégia", linewidth=2)       
    plt.plot(ibov.index, ibov["Ibov_cReturn"], label="Ibov Return", linewidth=2)
    plt.title("Comparação CSMOM e Ibov", fontsize=14)
    plt.xlabel("Data")
    plt.ylabel("Índice acumulado (base = 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
def graphs_bt(dfs, labels, ibov):

    plt.figure(figsize=(12,6))
    
    for i in range(0, len(dfs)):
        plt.plot(dfs[i].index, dfs[i]["winners_cum"], label="Estratégia {}".format(labels[i]), linewidth=2)       
    plt.plot(ibov.index, ibov["Ibov_cReturn"], label="Ibov Return", linewidth=2)
    plt.title("Comparação CSMOM e Ibov", fontsize=14)
    plt.xlabel("Data")
    plt.ylabel("Índice acumulado (base = 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def graph_bt_seaborn(bt, ibov):
    # Junta as séries
    df_plot = bt[["winners_cum"]].join(
        ibov[["Ibov_cReturn"]], how="inner"
    ).reset_index()

    df_plot.rename(columns={"index": "Date"}, inplace=True)

    # Formato tidy
    df_plot = df_plot.melt(
        id_vars="Date", 
        value_vars=["winners_cum", "Ibov_cReturn"], 
        var_name="Série", 
        value_name="Valor"
    )

    # Renomeando as legendas
    df_plot["Série"] = df_plot["Série"].map({
        "winners_cum": "Estratégia CSMOM",
        "Ibov_cReturn": "Ibovespa"
    })

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12,6))

    # Paleta manual: azul forte para a estratégia, cinza discreto para o Ibov
    palette = {
        "Estratégia CSMOM": "#7B68ee",  # azul forte
        "Ibovespa": "#b22222"           # cinza
    }

    sns.lineplot(
        data=df_plot, 
        x="Date", 
        y="Valor", 
        hue="Série", 
        linewidth=2.5, 
        palette=palette
    )

    plt.title("Comparação CSMOM e Ibov", fontsize=16, weight="bold")
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Índice acumulado (base = 1.0)", fontsize=12)
    plt.legend(title="Portfólios", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.show()
    
    

def graphs_bt_seaborn(dfs, labels, ibov):
    all_data = []
    
    # Estratégias
    for i in range(len(dfs)):
        temp = dfs[i][["winners_cum"]].copy()
        temp = temp.rename(columns={"winners_cum": "value"})
        temp["strategy"] = f"Estratégia {labels[i]}"
        temp["date"] = temp.index   # transforma o índice em coluna
        all_data.append(temp)
    
    # Ibov
    temp_ibov = ibov[["Ibov_cReturn"]].copy()
    temp_ibov = temp_ibov.rename(columns={"Ibov_cReturn": "value"})
    temp_ibov["strategy"] = "Ibov Return"
    temp_ibov["date"] = temp_ibov.index   # idem
    all_data.append(temp_ibov)

    # Concatena tudo
    plot_df = pd.concat(all_data)

    # Gráfico
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=plot_df, x="date", y="value", hue="strategy", linewidth=2)
    
    plt.title("Comparação CSMOM e Ibov", fontsize=14)
    plt.xlabel("Data")
    plt.ylabel("Índice acumulado (base = 1.0)")
    plt.grid(True, alpha=0.3)
    plt.show()
    

def plot_drawdown(df, label="Estratégia", color="lightcoral"):
    """
    Plota o drawdown de uma estratégia com área preenchida.
    
    df: DataFrame com coluna 'winners_cum'
    label: nome da estratégia
    color: cor do preenchimento da área
    """
    # Calcula drawdown
    drawdown = df["winners_cum"] / df["winners_cum"].cummax() - 1

    plt.figure(figsize=(12,6))
    plt.plot(df.index, drawdown, color='red', linewidth=2)
    plt.fill_between(df.index, drawdown, 0, color=color, alpha=0.5)

    plt.title("Drawdown da Estratégia", fontsize=14)
    plt.xlabel("Data")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.3)
    plt.show()


def simulation(lookback=12, skip=1,graph=True, save_csv=False, metrics=True, graph_maker="seaborn"):
    df = pd.read_csv("precos_b3_202010-2024_adjclose.csv")
    ibov_df_raw = pd.read_csv("ibov_2010_2024.csv")
    ibov_df = clean_df(ibov_df_raw)
    ibov_df = get_monthly_return(get_daily_return(ibov_df))
    ibov_df["Ibov_cReturn"] = (1.0 + ibov_df["Close"]).cumprod()
    ibov_df.rename(columns={'Close': 'Ibov_Return'}, inplace=True)
    prices = clean_df(df)
    bt = backtest_momentum_longshort(prices, lookback=lookback)
    bt = ibov_df.join(bt, how='inner')
    
    if graph:
        
        if graph_maker == "seaborn":
            graph_bt_seaborn(bt, ibov_df)
            plot_drawdown(bt)
        else:
            graph_bt(bt, ibov_df)
       
    
    if save_csv:
        bt.to_csv('ibov_x_momentum.csv', index=True)

    if metrics:
        metrics__year_df = calc_metrics_year(bt)
        metrics_df = calc_metrics_total(bt)
        
        metrics__year_df.to_csv("metrics_year_bt.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
        metrics_df.to_csv("metrics_total_bt.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
    
    return bt

def simulations(lookback=12, skip=1,graph=True, save_csv=False, metrics=True):
    df = pd.read_csv("precos_b3_202010-2024_adjclose.csv")
    ibov_df_raw = pd.read_csv("ibov_2010_2024.csv")
    ibov_df = clean_df(ibov_df_raw)
    ibov_df = get_monthly_return(get_daily_return(ibov_df))
    ibov_df["Ibov_cReturn"] = (1.0 + ibov_df["Close"]).cumprod()
    ibov_df.rename(columns={'Close': 'Ibov_Return'}, inplace=True)
    prices = clean_df(df)
    
    windows = [12, 9, 6, 3]
    labels = ["12-1", "9-1", "6-1", "3-1"]
    dfs = []
    
    for i in windows:
        bt = backtest_momentum_longshort(prices, lookback=i)
        dfs.append(bt)
        
    if graph:
        graphs_bt_seaborn(dfs, labels, ibov_df)
    
    
    return bt
    
simulation()
