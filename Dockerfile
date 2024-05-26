FROM python:3.8-slim

WORKDIR /usr/src/app
COPY . .
RUN pip install torch-2.2.1+cu118-cp38-cp38-linux_x86_64.whl
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "chat.py"]
