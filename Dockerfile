FROM python:3.11

RUN chmod 1777 /tmp

WORKDIR /workspace


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /workspace

EXPOSE 8000

# set environment variables
ENV OPENAI_API_KEY="sk-sduyrRdYYOdGP4x06e97DdDe7bA74c7e8a5aC1051d5a2831"
ENV OPENAI_BASE_URL="https://aihubmix.com/v1"

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
