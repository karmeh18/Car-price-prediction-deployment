# ============================================
# DOCKER COMMANDS FOR CAR PRICE PREDICTOR
# ============================================

# BUILDING THE IMAGE
# ==================

# Build image from Dockerfile
# Format: docker build -t <image-name>:<tag> <path-to-dockerfile>
# -t = tag (name) the image
docker build -t car-predictor:1.0 .

# Build with docker-compose (easier)
docker-compose build

# RUNNING THE CONTAINER
# =====================

# Run from image (basic)
docker run -p 9090:9090 car-predictor:1.0

# Run with interactive terminal (can see logs)
docker run -it -p 9090:9090 car-predictor:1.0

# Run with custom name
docker run -it -p 9090:9090 --name my-app car-predictor:1.0

# Run with docker-compose (RECOMMENDED)
docker-compose up

# Run detached (background)
docker-compose up -d

# USEFUL COMMANDS
# ===============

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# View container logs
docker logs <container-id-or-name>

# Follow logs in real-time
docker logs -f <container-id-or-name>

# Stop container
docker stop <container-id-or-name>

# Start stopped container
docker start <container-id-or-name>

# Remove container
docker rm <container-id-or-name>

# Remove image
docker rmi car-predictor:1.0

# Execute command in running container
docker exec -it <container-id> bash

# View image details
docker images

# DOCKER-COMPOSE COMMANDS
# =======================

# Start services (foreground)
docker-compose up

# Start services (background)
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and start
docker-compose up --build

# STEP-BY-STEP WORKFLOW
# ====================

# 1. Build the image
docker build -t car-predictor:1.0 .

# 2. Run the container
docker run -it -p 9090:9090 --name car-app car-predictor:1.0

# 3. Access the app
# Open browser: http://localhost:9090

# 4. View logs (in another terminal)
docker logs -f car-app

# 5. Stop the container
docker stop car-app

# 6. Restart it
docker start car-app

# 7. Clean up
docker rm car-app
docker rmi car-predictor:1.0
