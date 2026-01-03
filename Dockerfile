# Build Stage
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime Stage (Distroless for Security)
FROM python:3.9
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app
COPY fraud_model.joblib .
COPY app.py .

# Path for user-installed packages
ENV PATH=/root/.local/bin:$PATH

# Expose port 8000 for the banking API gateway
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]