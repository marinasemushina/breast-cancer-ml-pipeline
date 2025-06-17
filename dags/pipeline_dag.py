from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import config
from etl import load_data, preprocess, train_model, evaluate, save_results

def _load():
    return load_data.load_data()

def _preprocess(ti):
    path = ti.xcom_pull(task_ids='load_data')
    preprocess.preprocess(path)
    return path

def _train():
    input_path = os.path.join(config.RESULTS_DIR, config.DATA_FILE)
    train_model.train(input_path)

def _evaluate():
    input_path = os.path.join(config.RESULTS_DIR, config.DATA_FILE)
    evaluate.evaluate(input_path)

def _save_results():
    paths = [
        os.path.join(config.RESULTS_DIR, config.MODEL_FILE),
        os.path.join(config.RESULTS_DIR, config.METRICS_FILE)
    ]
    save_results.save_to_storage(paths)

with DAG(
    dag_id='ml_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    task1 = PythonOperator(
        task_id='load_data',
        python_callable=_load
    )

    task2 = PythonOperator(
        task_id='preprocess',
        python_callable=_preprocess
    )

    task3 = PythonOperator(
        task_id='train_model',
        python_callable=_train
    )

    task4 = PythonOperator(
        task_id='evaluate',
        python_callable=_evaluate
    )

    task5 = PythonOperator(
        task_id='save_results',
        python_callable=_save_results
    )

    task1 >> task2 >> task3 >> task4 >> task5
