# Step 1: Choose a base image (pre-built OS + Python)
FROM python:3.10-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy requirements.txt
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy entire project
COPY . .

# Step 6: Create artifacts directory structure
RUN mkdir -p /app/artifacts/reports && \
    mkdir -p /app/notebook && \
    mkdir -p /app/logs

# Step 7: Copy entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Step 8: Expose port
EXPOSE 9090

# Step 9: Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Step 10: Run entrypoint script (runs pipeline + app)
ENTRYPOINT ["/app/docker-entrypoint.sh"]
