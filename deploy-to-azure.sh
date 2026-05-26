#!/bin/bash
# Azure Deployment Script for Car Price Prediction ML Project
# Usage: bash deploy-to-azure.sh

set -e

echo "=================================="
echo "Azure Car Price Predictor Deploy"
echo "=================================="

# Configuration
RESOURCE_GROUP="car-prediction-rg"
LOCATION="eastus"
REGISTRY_NAME="carpredictionacr"
APP_NAME="car-price-predictor"
APP_PLAN="car-prediction-asp"
STORAGE_NAME="carpredictionstorage"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Login to Azure
echo -e "${BLUE}Step 1: Login to Azure...${NC}"
az login

# Step 2: Set subscription
echo -e "${BLUE}Step 2: Setting subscription...${NC}"
az account set --subscription "default"

# Step 3: Create Resource Group
echo -e "${BLUE}Step 3: Creating resource group...${NC}"
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Step 4: Create Container Registry
echo -e "${BLUE}Step 4: Creating Azure Container Registry...${NC}"
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $REGISTRY_NAME \
  --sku Basic \
  --admin-enabled true

# Get ACR details
ACR_LOGIN_SERVER=$(az acr show \
  --resource-group $RESOURCE_GROUP \
  --name $REGISTRY_NAME \
  --query loginServer \
  --output tsv)

echo -e "${GREEN}ACR Login Server: $ACR_LOGIN_SERVER${NC}"

# Step 5: Create App Service Plan
echo -e "${BLUE}Step 5: Creating App Service Plan...${NC}"
az appservice plan create \
  --name $APP_PLAN \
  --resource-group $RESOURCE_GROUP \
  --sku B1 \
  --is-linux

# Step 6: Create Web App
echo -e "${BLUE}Step 6: Creating Web App...${NC}"
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan $APP_PLAN \
  --name $APP_NAME \
  --deployment-container-image-name-user $ACR_LOGIN_SERVER/car-predictor:latest

# Step 7: Build and Push Docker Image
echo -e "${BLUE}Step 7: Building Docker image in ACR...${NC}"
az acr build \
  --registry $REGISTRY_NAME \
  --image car-predictor:latest \
  --file Dockerfile .

echo -e "${GREEN}✓ Image built and pushed${NC}"

# Step 8: Configure Web App with Docker Image
echo -e "${BLUE}Step 8: Configuring Web App with Docker image...${NC}"

USERNAME=$(az acr credential show \
  --name $REGISTRY_NAME \
  --query username \
  --output tsv)

PASSWORD=$(az acr credential show \
  --name $REGISTRY_NAME \
  --query passwords[0].value \
  --output tsv)

az webapp config container set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name $ACR_LOGIN_SERVER/car-predictor:latest \
  --docker-registry-server-url https://$ACR_LOGIN_SERVER \
  --docker-registry-server-username $USERNAME \
  --docker-registry-server-password $PASSWORD

# Step 9: Set Environment Variables
echo -e "${BLUE}Step 9: Setting environment variables...${NC}"
az webapp config appsettings set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    WEBSITES_ENABLE_APP_SERVICE_STORAGE=true

# Step 10: Create Storage Account
echo -e "${BLUE}Step 10: Creating Storage Account...${NC}"
az storage account create \
  --name $STORAGE_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

# Step 11: Create Blob Containers
echo -e "${BLUE}Step 11: Creating blob containers...${NC}"
for container in artifacts models reports logs; do
  az storage container create \
    --account-name $STORAGE_NAME \
    --name $container \
    --public-access off
done

# Step 12: Restart Web App
echo -e "${BLUE}Step 12: Restarting Web App...${NC}"
az webapp restart \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP

# Step 13: Get App URL
APP_URL=$(az webapp show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query defaultHostName \
  --output tsv)

echo ""
echo "=================================="
echo -e "${GREEN}✓ DEPLOYMENT COMPLETE!${NC}"
echo "=================================="
echo -e "${GREEN}Application URL: https://$APP_URL${NC}"
echo -e "${GREEN}Resource Group: $RESOURCE_GROUP${NC}"
echo -e "${GREEN}App Name: $APP_NAME${NC}"
echo -e "${GREEN}Container Registry: $ACR_LOGIN_SERVER${NC}"
echo ""
echo "Next steps:"
echo "1. Wait 2-3 minutes for container to start"
echo "2. Visit: https://$APP_URL"
echo "3. View logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo "4. Monitor: az monitor metrics list --resource /subscriptions/*/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/sites/$APP_NAME"
echo ""
