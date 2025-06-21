FROM upsidelab/enthusiast-server:latest

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential libpq-dev
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /build

COPY src/ ./

RUN poetry build -f wheel

RUN pip install --no-cache /build/dist/*.whl

WORKDIR /app

COPY config/settings_override.py ./pecl/settings_override.py

ENTRYPOINT ["/app/docker-entrypoint.sh"]

