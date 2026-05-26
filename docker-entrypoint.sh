#!/bin/bash
# Entrypoint script - runs pipeline then starts app

echo "=========================================="
echo "🐳 Docker Car Price Predictor"
echo "=========================================="

echo ""
echo "Step 1: Running Data Ingestion Pipeline..."
echo "  - Loading raw data"
echo "  - Splitting train/test"
echo "  - Transforming data"
echo "  - Training model"
echo ""

# Run data ingestion with automatic model comparison
# Use 'yes' as input to compare models by default in Docker
echo "yes" | python src/components/data_ingestion.py

echo ""
echo "=========================================="
echo "Step 2: Starting Flask Application..."
echo "=========================================="
echo "✅ App will run on: http://localhost:9090"
echo ""

# Start Flask app
python app.py
