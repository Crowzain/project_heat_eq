FROM python:3.11

WORKDIR /backend

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7080

CMD [ "streamlit", "run", "dashboard.py", "--server.port", "7080"]