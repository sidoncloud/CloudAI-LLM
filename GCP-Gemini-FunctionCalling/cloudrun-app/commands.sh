gcloud run deploy product-info-chatbot \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --max-instances 2 \
  --min-instances 1


curl -X POST https://product-info-chatbot-499192289487.us-central1.run.app/query \
-H "Content-Type: application/json" \
-d "{\"prompt\":\"Whats the cost of HDE Classic Bow Tie and how many orders were place for it in the past week\"}"

curl -X POST https://product-info-chatbot-499192289487.us-central1.run.app/query \
-H "Content-Type: application/json" \
-d "{\"prompt\":\"how many orders were placed for HDE Solid Color Suspenders - 20 Styles & HDE Classic Bow Tie?\"}"