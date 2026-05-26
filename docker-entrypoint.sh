#!/bin/bash
echo "=========================================="
echo "🐳 Docker Car Price Predictor"
echo "=========================================="
echo ""
echo "Checking model artifacts..."

# Check if model exists
if [ -f "/app/artifacts/model.pkl" ]; then
    echo "✅ model.pkl found"
else
    echo "⚠️ model.pkl not found - running training..."
    echo "yes" | python src/components/data_ingestion.py
fi

echo ""
echo "=========================================="
echo "Starting Flask Application..."
echo "=========================================="
echo "✅ App running on: http://localhost:9090"
echo ""

python app.py