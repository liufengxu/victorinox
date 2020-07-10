from collections import namedtuple
import pymysql
import logging
import pandas as pd

ConnConfig = namedtuple("ConnConfig", ['host', 'port', 'user', 'passwd', 'db'])
LOCAL_TEST = ConnConfig(host='127.0.0.1', port=3306, user='root'
                        , passwd='dnilqa0320', db='light_web')


def _connect(config):
    return pymysql.connect(host=config.host, port=config.port, user=config.user,
                           passwd=config.passwd, db=config.db, charset='utf8')


def query_sql(query_, config):
    mysql_cn = None

    try:
        mysql_cn = _connect(config)
        df_data = pd.read_sql(query_, con=mysql_cn)
        return df_data
    except Exception as e:
        logging.error("Exceptions happened in query_sql(): {0}".format(str(e)))
        raise
    finally:
        if mysql_cn is not None:
            mysql_cn.close()


def query_ad(query):
    return query_sql(query, LOCAL_TEST)

