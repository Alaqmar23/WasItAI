FROM python:3.10-slim

WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install dependencies (CPU versions of PyTorch to save massive amounts of Docker space)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY . .

# Hugging Face exposes port 7860 internally
EXPOSE 7860

# Run uvicorn server on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
