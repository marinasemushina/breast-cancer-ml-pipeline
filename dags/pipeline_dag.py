from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging
import os
import sys

# Добавляем путь к модулю etl
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'etl'))

# Импортируем функции из ETL-скриптов
from load_data import load_data_task
from preprocess import preprocess_task
from train_model import train_model_task
from evaluate import evaluate_task
from save_results import save_results_task

# Папка для логов
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='breast_cancer_pipeline',
    default_args=default_args,
    description='Pipeline для обработки данных и обучения модели по раку груди',
    schedule_interval=None,  # без расписания, запуск вручную
    catchup=False,
    tags=['ml', 'breast_cancer'],
) as dag:

    def log_wrapper(func):
        def wrapper(**context):
            log_file = os.path.join(LOG_DIR, f"{func.__name__}.log")
            handler = logging.FileHandler(log_file)
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            try:
                logger.info(f"Запуск задачи {func.__name__}")
                result = func(**context)
                logger.info(f"Задача {func.__name__} завершена успешно")
                return result
            except Exception as e:
                logger.error(f"Ошибка в задаче {func.__name__}: {e}")
                raise
            finally:
                logger.removeHandler(handler)
                handler.close()
        return wrapper

    task_load_data = PythonOperator(
        task_id='load_data',
        python_callable=log_wrapper(load_data_task),
        provide_context=True,
    )

    task_preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=log_wrapper(preprocess_task),
        provide_context=True,
    )

    task_train_model = PythonOperator(
        task_id='train_model',
        python_callable=log_wrapper(train_model_task),
        provide_context=True,
    )

    task_evaluate = PythonOperator(
        task_id='evaluate',
        python_callable=log_wrapper(evaluate_task),
        provide_context=True,
    )

    task_save_results = PythonOperator(
        task_id='save_results',
        python_callable=log_wrapper(save_results_task),
        provide_context=True,
    )

    # Определяем зависимости
    task_load_data >> task_preprocess >> task_train_model >> task_evaluate >> task_save_results
