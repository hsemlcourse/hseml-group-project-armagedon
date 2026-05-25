[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание наличия диабета

**Студент:** Курило Илья Константинович

**Группа:** БИВ231


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Запуск сервиса](#запуск-сервиса)
3. [API](#api)
4. [Результаты](#результаты)
5. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Классификация (диабет есть/нет)

**Датасет:** [Diabetes Health Indicators Dataset (diabetes _ 012 _ health _ indicators _ BRFSS2015.csv)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)

**Целевая метрика:** Precision, Recall, F1-score, ROC-AUC, PR-AUC


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── api.py                  # FastAPI и HTML-интерфейс
│   └── modeling.py             # Обучение и экспорт финальной модели
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск сервиса

```bash
# Сервис с готовой моделью
docker compose up --build
```

После запуска:

- интерфейс: <http://localhost:8000/>
- Swagger/OpenAPI: <http://localhost:8000/docs>
- проверка состояния: <http://localhost:8000/health>

Если требуется переобучить артефакт по исходному CSV:

```bash
docker compose run --rm ml-project python -m src.modeling
```

## API

`POST /predict` принимает 21 показатель анкеты BRFSS и возвращает вероятность
принадлежности к группе риска и бинарный прогноз. Например:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"HighBP":1,"HighChol":1,"CholCheck":1,"BMI":33,"Smoker":0,"Stroke":0,"HeartDiseaseorAttack":0,"PhysActivity":0,"Fruits":1,"Veggies":1,"HvyAlcoholConsump":0,"AnyHealthcare":1,"NoDocbcCost":0,"GenHlth":4,"MentHlth":2,"PhysHlth":5,"DiffWalk":0,"Sex":1,"Age":9,"Education":5,"Income":6}'
```

## Результаты

| Модель | F1-Score | ROC-AUC | Примечание |
|--------|-------------|-------------|------------|
| Baseline (Logistic Regression) | 0.4700 | 0.8177 | Хорошие линейные зависимости |
| CatBoost (Лучшая) | 0.4696 | 0.7432 | |
| LightGBM (Tuned) | 0.4686 | 0.7468 | |
| XGBoost | 0.4671 | 0.7416 | |
| Stacking Ensemble | 0.3157 | 0.5937 | Склонность к переобучению на мажоритарный класс |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
