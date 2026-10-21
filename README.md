# Лаба 6. Docker и обучение в контейнере

Нейроранкер для рекомендаций обучается на синтетических взаимодействиях user × item (`data/interactions.csv`). Нужно упаковать обучение в Docker-образ, подключить его к MLflow, проверить продолжение с checkpoint, передать образ через registry и запустить тот же код на GPU в Colab.

Шаги 1–8 выполняются на вашем компьютере на CPU, шаг 9 в Google Colab. Аккаунты, кроме Google, не нужны.

| Файл | Что это |
|---|---|
| `src/train.py` | Обучение: `--epochs --lr --batch-size --resume`, checkpoint в `/outputs/ckpt.pt` |
| `data/interactions.csv` | Данные: 4000 пользователей, 2000 объектов, 240 000 взаимодействий |
| `requirements.txt` | Зависимости обучения поверх базового образа |
| `Dockerfile.base` | Базовый образ `lab-base:1`: Python 3.12 и PyTorch для CPU |
| `compose.yaml`, `.env`, `mlflow/` | MLflow-стек: PostgreSQL, RustFS (S3), сервер MLflow |
| `colab/step9_gpu.ipynb` | Шаг 9 |

## Шаг 1. Подготовка

Команды скачивают около 3 ГБ образов и пакетов.

1. Установите Docker Desktop (macOS) или Docker Engine (Linux).
2. Проверьте установку: `docker run hello-world`.
3. В папке `lab/` выполните:
   ```bash
   docker pull python:3.12-slim
   docker build -f Dockerfile.base -t lab-base:1 .
   docker compose build
   docker compose pull --ignore-buildable
   docker pull registry:3
   ```

**Проверка:** `docker images` показывает `lab-base:1` и `lab-mlflow:3.16.1`.

## Шаг 2. Контейнер руками и `docker commit`

`docker commit` сохраняет слой записи контейнера как новый образ. `docker diff` показывает изменения в этом слое: `A` — файл добавлен, `C` — изменён.

```bash
docker run -it --name scratch python:3.12-slim bash
# внутри контейнера:
pip install requests
echo "ручная правка" > /opt/note.txt
exit
```

```bash
docker diff scratch | grep -v /root/.cache | head -15
docker commit scratch scratch:manual
docker run --rm scratch:manual cat /opt/note.txt
docker history scratch:manual
```

**Проверка:** `cat` выводит «ручная правка», верхний слой в `docker history` создан командой `bash`.

**Вопрос:** как коллега узнает, что установлено в `scratch:manual`, и как соберёт такой же образ на Python 3.13?

## Шаг 3. Dockerfile

1. Напишите `Dockerfile` в корне `lab/`:
   - базовый образ `lab-base:1`;
   - рабочий каталог `/app`;
   - сначала скопировать `requirements.txt` и установить зависимости, потом скопировать `src/`;
   - `ENV PYTHONUNBUFFERED=1`;
   - `ENTRYPOINT` в JSON-форме запускает `python src/train.py`, `CMD` задаёт `--epochs 10`.
2. Создайте `.dockerignore`: `data/`, `outputs/`, `colab/`, `mlflow/`, `.git`, `.env`.
3. Соберите образ: `docker build -t ranker:v1 .`
4. Измените текст любого `print` в `src/train.py` и соберите снова.

**Проверка:** во второй сборке шаг `RUN pip install` помечен `CACHED`, заново выполняется только `COPY src/`.

**Вопрос:** что произойдёт с кэшем, если `COPY src/` поставить перед `RUN pip install`? Проверьте.

## Шаг 4. Каталоги хоста и аргументы

Команды в этом шаге составьте сами. Аргументы `train.py` выводит `docker run --rm ranker:v1 --help`, флаги Docker выводит `docker run --help`. Каталог для результатов создайте заранее: `mkdir -p outputs`.

1. Запустите обучение на 3 эпохи с `--lr 1e-3`. Контейнер должен видеть `./data` только для чтения, checkpoint должен попасть в `./outputs` на хосте, а сам контейнер должен удалиться после завершения.
2. Запустите `train.py` с флагом `--write-test` и с тем же подключением `./data`.
3. Запустите контейнер из `ranker:v1` так, чтобы вместо обучения открылась оболочка `bash`, и посмотрите содержимое `/app`.

**Проверка:** в логе три строки `epoch`, в `outputs/` на хосте появился `ckpt.pt`. `--write-test` завершается ошибкой `Read-only file system`. В `/app` лежат `requirements.txt` и `src`.

**Вопрос:** куда пропадёт `ckpt.pt`, если запустить контейнер без `-v ./outputs:/outputs`?

## Шаг 5. MLflow в Docker Compose

```bash
docker compose up -d
docker compose ps
```

Откройте http://localhost:5050. Затем запустите обучение в сети Compose:

```bash
docker run --rm --network lab_default \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -v ./data:/data:ro -v ./outputs:/outputs \
  ranker:v1 --epochs 10
```

**Проверка:** в эксперименте `ranker` есть run с графиками `loss` и `recall_at_10`, во вкладке Artifacts лежит `ckpt.pt`.

Остановите стек командой `docker compose down` и поднимите снова: run на месте. Команда `docker compose down -v` удаляет тома вместе с run, её не выполняйте до конца лабы.

**Вопрос:** почему обучение обращается к `http://mlflow:5000`, а браузер к `localhost:5050`?

## Шаг 6. Фоновый запуск, остановка, продолжение

```bash
docker run -d --name exp1 --network lab_default \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -v ./data:/data:ro -v ./outputs:/outputs \
  ranker:v1 --epochs 30
docker logs -f exp1
```

`Ctrl+C` отключает только просмотр лога, обучение продолжается. Через несколько эпох:

```bash
docker stop exp1
docker logs exp1 | tail -3
docker rm exp1
docker run -d --name exp1 --network lab_default \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -v ./data:/data:ro -v ./outputs:/outputs \
  ranker:v1 --epochs 30 --resume
docker logs exp1 | grep resumed
```

**Проверка:** перед остановкой в логе есть `SIGTERM received, saving checkpoint`, после запуска `resumed from epoch N`. В MLflow график того же run продолжается с эпохи N + 1.

## Шаг 7. Лимит памяти

```bash
docker run --rm --memory 512m -v ./data:/data:ro ranker:v1 --epochs 1
docker run --name oom --memory 512m -v ./data:/data:ro ranker:v1 --batch-size 2000000
docker inspect oom --format '{{.State.ExitCode}} {{.State.OOMKilled}}'
docker rm oom
```

**Проверка:** с обычным батчем эпоха проходит в 512 МБ. С батчем на весь датасет `docker inspect` выводит `137 true`: ядро завершило процесс, Python не успел вывести ошибку.

## Шаг 8. Registry

Запустите локальный registry:

```bash
docker run -d -p 5001:5000 --name registry registry:3
```

Дальше команды составьте сами: понадобятся `docker tag`, `docker push`, `docker pull` и `docker image rm`.

1. Отправьте `ranker:v1` в registry под именем `localhost:5001/ranker:v1`. Запишите digest из вывода `docker push`.
2. Удалите локальные образы `localhost:5001/ranker:v1` и `ranker:v1`, затем скачайте образ из registry.
3. Измените текст первого `print` в `src/train.py`, соберите образ и отправьте его в registry под тем же тегом `localhost:5001/ranker:v1`.
4. Скачайте первую версию по digest из пункта 1 и запустите обе версии на одну эпоху.

**Проверка:** `docker images --digests localhost:5001/ranker` показывает два разных digest. Версия по digest печатает старый текст, версия по тегу `v1` печатает новый.

**Вопрос:** какую ссылку на образ вы запишете в скрипт запуска обучения на сервере, тег или digest, и почему?

## Шаг 9. Тот же код на GPU в Colab

Откройте ноутбук в Google Colab по ссылке https://colab.research.google.com/github/dslvt/ml-docker-lab/blob/main/colab/step9_gpu.ipynb или загрузите `colab/step9_gpu.ipynb` через «Файл → Загрузить блокнот» и следуйте инструкциям. Ячейки 1–3 запустите в начале лабы: они скачивают образ PyTorch с CUDA и работают 5–10 минут.

**Проверка:** в выводе обучения `device=cuda`, после `--resume` строка `resumed from epoch 5`.

## Сдача

1. Вывод `docker images --digests localhost:5001/ranker` из шага 8.
2. Скриншот run в MLflow с графиком `recall_at_10` после продолжения из шага 6.
3. Строка `resumed from epoch N` из шага 6.
4. Если дошли до шага 9: строки `epoch` с `device=cuda` из Colab.

## Уборка после лабы

```bash
docker compose down -v
docker rm -f registry scratch
docker image rm scratch:manual
docker image rm -f $(docker images -q localhost:5001/ranker)
```
