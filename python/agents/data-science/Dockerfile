FROM python:3.12-slim

# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip
# Install uv
RUN pip install --no-cache-dir uv>=0.7.19

# Export requirements to a temporary file and install with pip
# This avoids 'uv sync' authentication issues with internal mirrors
COPY uv.lock pyproject.toml README.md *.json ./
RUN uv export --frozen --no-dev --no-hashes --output-file requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8080

CMD ["python", "demo/app.py"]
