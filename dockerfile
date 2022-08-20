FROM python:3.9

ENV PIP_DISABLE_PIP_VERSION_CHECK=on

RUN pip install poetry

WORKDIR /app
COPY ./ /app/

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction

RUN ["chmod", "+x", "run.sh"]
CMD [ "./run.sh" ]