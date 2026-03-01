# Azure Deployment Documentation

## Deployment Details

**App URL**: https://meerkat.azurewebsites.net
**Status**: Running
**Region**: Central India
**Tier**: B2 Basic (~$26/month)
**Resource Group**: meerkat-prod-rg
**App Service Plan**: meerkat-prod-plan

## Environment Variables

The following environment variables are configured:
- `OPENAI_API_KEY` - OpenAI API key for document processing
- `FIREBASE_SERVICE_ACCOUNT_PATH` - Path to Firebase service account key
- `RAZORPAY_KEY_ID` - Razorpay payment gateway key
- `RAZORPAY_KEY_SECRET` - Razorpay secret key
- `RAZORPAY_WEBHOOK_SECRET` - Razorpay webhook verification secret
- `REPORT_PRICE_PAISE` - Report price in paise (99900 = ₹999)
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
az webapp log tail --resource-group meerkat-prod-rg --name meerkat
```

### Download Logs
```bash
az webapp log download --resource-group meerkat-prod-rg --name meerkat --log-file app-logs.zip
```

### Restart App
```bash
az webapp restart --resource-group meerkat-prod-rg --name meerkat
```

### Update Environment Variables
```bash
az webapp config appsettings set --resource-group meerkat-prod-rg --name meerkat --settings KEY=VALUE
```

### Deploy Updates
```bash
az webapp up --resource-group meerkat-prod-rg --name meerkat --plan meerkat-prod-plan --runtime "PYTHON:3.11" --sku B2 --location centralindia
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
curl https://meerkat.azurewebsites.net/healthz
```

### View in Azure Portal
https://portal.azure.com > Resource Groups > meerkat-prod-rg > meerkat

## Cost Management

**B2 Basic Tier Costs**:
- ~$26/month (Pay as you go)
- 2 cores, 3.5 GB RAM

### Stop App to Save Costs
```bash
az webapp stop --resource-group meerkat-prod-rg --name meerkat
```

### Delete Resources
```bash
az group delete --name meerkat-prod-rg --yes
```

## Troubleshooting

### Common Issues

**503 Service Unavailable**: App is starting or crashed
- Check logs: `az webapp log tail --resource-group meerkat-prod-rg --name meerkat`
- Restart: `az webapp restart --resource-group meerkat-prod-rg --name meerkat`

**Module Not Found Errors**: Dependencies not installed
- Redeploy with: `az webapp up --resource-group meerkat-prod-rg --name meerkat --runtime "PYTHON:3.11"`

**Timeout Errors**: Gunicorn timeout (currently set to 600s)
- Increase in startup command if needed

## Security Notes

- HTTPS is enabled by default
- API keys are stored securely in App Settings
- Temporary files are stored in `/tmp` and cleaned on restart
- No persistent storage configured (ephemeral file system)
