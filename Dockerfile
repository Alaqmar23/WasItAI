FROM python:3.10-slim

WORKDIR /code

# Copy specific backend files to the container
COPY ./backend/requirements.txt /code/requirements.txt

# Install dependencies (CPU versions to save space)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the backend code
COPY ./backend /code

# Hugging Face default port
EXPOSE 7860

# Run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
