import pandas as pd
from clickhouse_driver import Client
from tqdm import tqdm
import logging
from config.config import (
    CH_HOST, CH_USER, CH_PASSWORD, CH_DATABASE, CHECKPOINT_DATA_PATH
)
from utils.checkpoint import save_data_processing_checkpoint, load_data_processing_checkpoint

def connect_clickhouse():
    try:
        client = Client(host=CH_HOST, user=CH_USER, password=CH_PASSWORD, database=CH_DATABASE)
        logging.info("Conexão ao ClickHouse estabelecida com sucesso.")
        return client
    except Exception as e:
        logging.error(f"Erro ao conectar ao ClickHouse: {e}")
        raise

def get_tick_bars_in_chunks(client, chunk_size, total_estimate, last_bar_time=None):
    try:
        with tqdm(total=total_estimate, desc="Fetching tick bars", unit=" records") as pbar:
            while True:
                if last_bar_time is None:
                    query = f'''
                    SELECT
                        bar_time,
                        close,
                        high,
                        low,
                        volume
                    FROM tick_bars
                    ORDER BY bar_time
                    LIMIT {chunk_size}
                    '''
                else:
                    query = f'''
                    SELECT
                        bar_time,
                        close,
                        high,
                        low,
                        volume
                    FROM tick_bars
                        WHERE bar_time > '{last_bar_time}'
                    ORDER BY bar_time
                    LIMIT {chunk_size}
                    '''
                data = list(client.execute_iter(query))
                if not data:
                    break

                if len(data[0]) != 5:
                    logging.warning("Formato inesperado dos dados. Verifique a consulta SQL.")
                    print("Formato inesperado dos dados. Verifique a consulta SQL.")
                    break

                df_chunk = pd.DataFrame(data, columns=['bar_time', 'close', 'high', 'low', 'volume'])
                df_chunk['bar_time'] = pd.to_datetime(df_chunk['bar_time'])
                yield df_chunk
                last_bar_time = data[-1][0]
                pbar.update(len(data))
                
                # Salvar o checkpoint após processar cada bloco
                save_data_processing_checkpoint(last_bar_time)
    except Exception as e:
        logging.error(f"Erro durante a recuperação de dados: {e}")
        raise
