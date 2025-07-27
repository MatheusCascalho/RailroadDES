# config_log.py
import logging
from datetime import datetime

def configurar_logs():
    # Configuração básica de logging
    t = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    logging.basicConfig(level=logging.INFO,  # Define o nível mínimo de log
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),  # Exibe no terminal
                            logging.FileHandler(f'logs_treinamento_{t}.log', 'w', 'utf-8')  # Salva em arquivo
                        ])

class CriticalFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.CRITICAL

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

def separated_logs():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    critical_handler = logging.FileHandler('training_logs.log', 'w', 'utf-8')
    critical_handler.setLevel(logging.CRITICAL)
    critical_handler.addFilter(CriticalFilter())
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    critical_handler.setFormatter(formatter)

    info_filter = logging.FileHandler('training_events.log', 'w', 'utf-8')
    info_filter.setLevel(logging.INFO)
    info_filter.addFilter(InfoFilter())
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    info_filter.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)
    console_handler.addFilter(CriticalFilter())


    logger.addHandler(critical_handler)
    logger.addHandler(info_filter)
    logger.addHandler(console_handler)


# Chama a função de configuração de logs
configurar_logs()
# separated_logs()