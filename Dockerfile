FROM python:3.10-slim

WORKDIR /code

# Copy requirements from the backend folder
COPY ./backend/requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the backend files
COPY ./backend /code

# Expose the standard Hugging Face port
EXPOSE 7860

# Start the FastApi server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
