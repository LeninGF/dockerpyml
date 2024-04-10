from sqlalchemy import text, create_engine

def conectar_sql(big_data_bbdd=True, db_user='falconiel', analitica_user_password='BebuSuKO', proxy_user_password='N27a34v1', analitica_host='192.168.152.197', proxy_host='192.168.152.8'):
    
    if big_data_bbdd:
        engine_maria_db = create_engine(f"mysql+pymysql://{db_user}:{analitica_user_password}@{analitica_host}", pool_recycle=3600)
        print(f"conectando {db_user}@{analitica_host}. Espere por favor...")
    else:
        # F0s!Hu63
        engine_maria_db = create_engine(f"mysql+pymysql://{db_user}:{proxy_user_password}@{proxy_host}", pool_recycle=3600)
        print(f"conectando {db_user}@{proxy_host}. Espere por favor...")
    print(engine_maria_db.connect())
    return engine_maria_db