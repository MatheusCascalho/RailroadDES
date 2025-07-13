# config_log.py
import logging

def configurar_logs():
    # Configuração básica de logging
    logging.basicConfig(level=logging.CRITICAL,  # Define o nível mínimo de log
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),  # Exibe no terminal
                            logging.FileHandler('meu_projeto_2.log', 'w', 'utf-8')  # Salva em arquivo
                        ])

# Chama a função de configuração de logs
configurar_logs()