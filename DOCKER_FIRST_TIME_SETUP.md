# 🐳 Docker First-Time Setup - Complete Flow

## What Happens When Docker Container Starts?

```
1. Docker reads Dockerfile
2. Builds image with Python + dependencies
3. Copies entire project into container
4. Container STARTS
5. Runs docker-entrypoint.sh script
   ↓
   a) Executes: python src/components/data_ingestion.py
      - Reads notebook\car_data.csv
      - Splits into train/test
      - Creates preprocessor.pkl
      - Trains model
      - Saves model.pkl
   ↓
   b) Executes: python app.py
      - Flask server starts
      - Listens on port 9090
      - Ready for predictions!
```

---

## Step-by-Step: First Time With Docker

### Step 1: Build Docker Image
```powershell
cd d:\car_prediction_project
docker build -t car-predictor:1.0 .
```

**What this does:**
- Reads Dockerfile
- Downloads Python 3.10 base image
- Installs all dependencies from requirements.txt
- Copies project files
- Creates image: ~1-2GB
- **Time: 5-10 minutes first time**

### Step 2: Run Docker Container
```powershell
docker run -it -p 9090:9090 --name car-app car-predictor:1.0
```

**What this does:**
```
Container starts...
  ↓
Runs docker-entrypoint.sh
  ↓
Executes: python src/components/data_ingestion.py
  Loading data...
  Splitting data... (70% train, 30% test)
  Transforming data...
  Training model... (takes 3-5 minutes)
  ✅ Complete!
  ↓
Executes: python app.py
  ✅ Flask running on http://0.0.0.0:9090
```

### Step 3: Test Application
```
Open browser: http://localhost:9090
Fill form → Click "Predict Price"
See prediction!
```

### Step 4: Stop Container
```powershell
# In another PowerShell window:
docker stop car-app

# Or press Ctrl+C in container window
```

---

## Using Docker-Compose (Easier)

```powershell
# Build and run with one command
docker-compose up --build

# Stop
docker-compose down
```

---

## Files Explanation

### `Dockerfile` Changes
```dockerfile
# NEW: Copy entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# CHANGED: Use ENTRYPOINT instead of CMD
# CMD only runs the app
# ENTRYPOINT runs the script that does everything
ENTRYPOINT ["/app/docker-entrypoint.sh"]
```

### `docker-entrypoint.sh` (New File)
```bash
#!/bin/bash
# This script runs when container starts

# Step 1: Run pipeline
python src/components/data_ingestion.py

# Step 2: Run Flask app
python app.py
```

---

## First-Time Timeline

| Step | Time | What Happens |
|------|------|--------------|
| `docker build` | 5-10 min | Downloads Python, installs packages |
| `docker run` | ~1-2 min | Container starts |
| Data ingestion | ~5-10 min | Loads, splits, transforms, trains |
| Flask starts | ~10 sec | Ready to use |
| **Total** | **~15-25 min** | **Complete setup!** |

---

## Advantages of Docker First-Time Setup

| Aspect | Without Docker | With Docker |
|--------|---------------|------------|
| **Setup steps** | 15+ commands | 1 command |
| **Dependency issues** | Scikit-learn version mismatch | None (locked) |
| **Works on other machines** | Maybe (probably not) | Guaranteed |
| **Time to deploy** | 30+ minutes | 2 minutes |
| **Reproducibility** | Different every time | Always same |

---

## Architecture Flow

```
Your Computer
    ↓
docker build
    ↓
Creates: car-predictor:1.0 (image)
    ↓
docker run
    ↓
Creates: car-app (container)
    ↓
Container runs docker-entrypoint.sh
    ↓
├─ python src/components/data_ingestion.py
│   ├─ Load notebook\car_data.csv
│   ├─ Split train/test
│   ├─ Transform data
│   ├─ Train model
│   ├─ Save preprocessor.pkl
│   └─ Save model.pkl
│
└─ python app.py
    └─ Flask listening on port 9090
        └─ http://localhost:9090
```

---

## Complete Commands Reference

### First Time (Local Machine)
```powershell
# 1. Build image
docker build -t car-predictor:1.0 .

# 2. Run container
docker run -it -p 9090:9090 --name car-app car-predictor:1.0

# Output will show:
# Step 1: Data Ingestion...
# Training model... (wait 3-5 min)
# Step 2: Flask running on http://0.0.0.0:9090

# 3. Open browser: http://localhost:9090
```

### Using Docker-Compose (Simpler)
```powershell
# Build + Run
docker-compose up --build

# Run (already built)
docker-compose up

# Stop
docker-compose down

# View logs
docker-compose logs -f
```

### On Another Machine (Deployment)
```bash
# 1. Only need image (no build needed!)
docker pull your-registry/car-predictor:1.0

# 2. Run
docker run -it -p 9090:9090 your-registry/car-predictor:1.0

# Same result: pipeline runs, app starts!
```

---

## What Gets Created Inside Container

```
/app (inside container)
├── src/
├── Pipeline/
├── templates/
├── artifacts/
│   ├── train.csv         ← Created by data_ingestion.py
│   ├── test.csv          ← Created by data_ingestion.py
│   ├── preprocessor.pkl  ← Created by transformation
│   └── model.pkl         ← Created by training
├── requirements.txt
├── app.py
├── Dockerfile
└── docker-entrypoint.sh
```

---

## FAQ

### Q: Does it run data_ingestion.py every time?
**A:** Yes, every time you start the container:
- Good: Always has fresh data
- Bad: Takes 5-10 minutes each time

**Solution for production:** Create a separate image with pre-built artifacts:
```dockerfile
# Production version: skip data_ingestion
CMD ["python", "app.py"]
```

### Q: Can I skip data_ingestion?
**A:** Yes, modify docker-entrypoint.sh:
```bash
# Comment out this line:
# python src/components/data_ingestion.py

# Just run:
python app.py
```

### Q: What if container crashes?
**A:** Check logs:
```powershell
docker logs car-app
```

### Q: How to restart?
```powershell
docker restart car-app
```

---

## Summary

**Docker First-Time Flow:**
1. Build image once: `docker build -t car-predictor:1.0 .`
2. Run container: `docker run -it -p 9090:9090 car-predictor:1.0`
3. Container automatically:
   - Runs data_ingestion.py (5-10 min)
   - Trains model
   - Starts Flask app
4. Open browser → Use app!

**Advantages:**
- ✅ Reproducible everywhere
- ✅ No manual setup needed
- ✅ All dependencies included
- ✅ Easy deployment

---
