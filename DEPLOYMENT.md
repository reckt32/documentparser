# Azure Deployment Documentation

## Deployment Details

**App URL**: https://docparser-app.azurewebsites.net
**Status**: Running
**Region**: Central India
**Tier**: B1 Basic (~$13/month)
**Resource Group**: docparser-prod-rg
**App Service Plan**: docparser-prod-plan

## Environment Variables

The following environment variables are configured:
- `OPENAI_API_KEY` - OpenAI API key for document processing
- `STORE_REPORTS` - Set to `memory`
- `OUTPUT_DIR` - Set to `/tmp`
- `SAVE_TX_JSON` - Set to `false`
- `WEBSITES_PORT` - Set to `8000`
- `ENABLE_ORYX_BUILD` - Set to `true`

## API Endpoints

- `GET /healthz` - Health check endpoint
- `POST /login` - User authentication
- `POST /upload` - Upload documents for parsing
- `GET /download/<filename>` - Download processed files
- `GET /download-temp/<token>` - Download temporary files
- `POST /questionnaire/start` - Start new questionnaire
- `PUT /questionnaire/<id>/<section>` - Update questionnaire section
- `GET /questionnaire/<id>` - Get questionnaire data
- `POST /report/generate` - Generate financial report

## Management Commands

### View Logs
```bash
az webapp log tail --resource-group docparser-prod-rg --name docparser-app
```

### Download Logs
```bash
az webapp log download --resource-group docparser-prod-rg --name docparser-app --log-file app-logs.zip
```

### Restart App
```bash
az webapp restart --resource-group docparser-prod-rg --name docparser-app
```

### Update Environment Variables
```bash
az webapp config appsettings set --resource-group docparser-prod-rg --name docparser-app --settings KEY=VALUE
```

### Deploy Updates
```bash
az webapp up --resource-group docparser-prod-rg --name docparser-app --plan docparser-prod-plan --runtime "PYTHON:3.11" --sku B1 --location centralindia
```

## Deployment Architecture

- **Runtime**: Python 3.11
- **Web Server**: Gunicorn
- **Build System**: Oryx (Azure's build automation)
- **Virtual Environment**: Located at `/tmp/<build-id>/antenv`
- **Dependencies**: Installed from `requirements.txt`

## Monitoring

### Check App Status
```bash
curl https://docparser-app.azurewebsites.net/healthz
```

### View in Azure Portal
https://portal.azure.com > Resource Groups > docparser-prod-rg > docparser-app

## Cost Management

**B1 Basic Tier Costs**:
- ~$13.14/month (Pay as you go)
- 1 core, 1.75 GB RAM
- Free tier available: Azure for Students includes credits

### Stop App to Save Costs
```bash
az webapp stop --resource-group docparser-prod-rg --name docparser-app
```

### Delete Resources
```bash
az group delete --name docparser-prod-rg --yes
```

## Troubleshooting

### Common Issues

**503 Service Unavailable**: App is starting or crashed
- Check logs: `az webapp log tail --resource-group docparser-prod-rg --name docparser-app`
- Restart: `az webapp restart --resource-group docparser-prod-rg --name docparser-app`

**Module Not Found Errors**: Dependencies not installed
- Redeploy with: `az webapp up --resource-group docparser-prod-rg --name docparser-app --runtime "PYTHON:3.11"`

**Timeout Errors**: Gunicorn timeout (currently set to 600s)
- Increase in startup command if needed

## Security Notes

- HTTPS is enabled by default
- OPENAI_API_KEY is stored securely in App Settings
- Temporary files are stored in `/tmp` and cleaned on restart
- No persistent storage configured (ephemeral file system)

## Deployment Date
November 29, 2025
