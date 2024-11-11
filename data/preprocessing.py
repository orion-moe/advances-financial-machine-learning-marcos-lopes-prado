import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import comb
from numba import njit, prange
import logging
from utils.checkpoint import append_removed_records

@njit(parallel=True)
def triple_barrier_numba(close_prices, pt_sl, max_hold):
    N = len(close_prices)
    labels = np.zeros(N, dtype=np.int8)  # 0: neutral, 1: take profit, -1: stop loss

    for i in prange(N):
        entry_price = close_prices[i]
        tp = entry_price * (1 + pt_sl[0])
        sl = entry_price * (1 - pt_sl[1])
        exit_idx = i + max_hold if (i + max_hold) < N else N
        flag = 0

        for j in range(i + 1, exit_idx):
            price = close_prices[j]
            if price >= tp:
                flag = 1
                break
            elif price <= sl:
                flag = -1
                break
        labels[i] = flag

    return labels

def triple_barrier_method(df, pt_sl=(0.02, 0.02), max_hold=10):
    close_prices = df['close'].values
    labels = triple_barrier_numba(close_prices, pt_sl, max_hold)
    df['label'] = labels
    return df

def verify_data_quality(df, removed_records):
    try:
        duplicates = df[df.duplicated(subset='bar_time', keep=False)]
        df = df.drop_duplicates(subset='bar_time')

        na = df[df.isna().any(axis=1)]
        df = df.dropna()

        z_scores = np.abs(stats.zscore(df[['close', 'high', 'low', 'volume']]))
        outliers = df[(z_scores >= 3).any(axis=1)]
        df = df[(z_scores < 3).all(axis=1)]

        # Salvar registros removidos
        removed_data = pd.concat([duplicates, na, outliers]).drop_duplicates()
        removed_records.append(removed_data)

        return df
    except Exception as e:
        logging.error(f"Erro durante a verificação da qualidade dos dados: {e}")
        raise

def calculate_vwap(df):
    try:
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    except Exception as e:
        logging.error(f"Erro durante o cálculo do VWAP: {e}")
        raise

def calculate_moving_averages(df, windows=[5, 10, 20]):
    try:
        for window in windows:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        return df
    except Exception as e:
        logging.error(f"Erro durante o cálculo das médias móveis: {e}")
        raise

def fractional_difference(series, d=0.5, thresh=1e-5):
    """
    Calcula a diferenciação fracionária de uma série temporal.
    """
    try:
        def frac_diff_coeff(d, k):
            return comb(d, k, exact=False)

        max_lag = 100  # Limite para o cálculo
        frac_diff = [1.0]
        for k in range(1, max_lag):
            coeff = frac_diff_coeff(d, k)
            frac_diff.append(coeff)
            if abs(coeff) < thresh:
                break

        frac_diff = np.array(frac_diff).reshape(-1, 1)
        df_frac = pd.Series(0, index=series.index)
        for i in range(len(frac_diff)):
            df_frac += frac_diff[i] * series.shift(i)

        df_frac = df_frac.dropna()
        return df_frac
    except Exception as e:
        logging.error(f"Erro durante a diferenciação fracionária: {e}")
        raise

def process_batch(df, all_data, removed_records):
    try:
        df = verify_data_quality(df, removed_records)
        df = df.sort_values('bar_time').reset_index(drop=True)
        df = calculate_vwap(df)
        df = calculate_moving_averages(df)
        all_data.append(df)
        return df
    except Exception as e:
        logging.error(f"Erro durante o processamento do lote de dados: {e}")
        raise
