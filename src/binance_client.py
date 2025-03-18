from binance.client import Client
import os
from .config import load_env, get_env_variable

def get_binance_client():
    load_env()
    api_key = get_env_variable('BINANCE_API_KEY')
    api_secret = get_env_variable('BINANCE_API_SECRET')
    client = Client(api_key, api_secret)
    return client
