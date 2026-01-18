---
sidebar_position: 8
---

# API Usage

Complete documentation for the Sentiment Analysis REST API endpoints with example requests.

## Base URL

```
http://localhost:8000
```

For production deployments, replace with your deployed URL.

## Authentication

Currently, the API does not require authentication. For production, implement API key or OAuth2 authentication.

## Endpoints

### Health Check

Check the status of the API and loaded models.

```http
GET /healthcheck
```

#### Response

```json
{
  "status": "ok",
  "ml_model": true,
  "dl_model": true
}
```

| Field      | Type    | Description                  |
| ---------- | ------- | ---------------------------- |
| `status`   | string  | API status ("ok" or "error") |
| `ml_model` | boolean | Whether ML model is loaded   |
| `dl_model` | boolean | Whether DL model is loaded   |

#### Example

```bash
curl http://localhost:8000/healthcheck
```

---

### Predict with ML Model

Analyze sentiment using the Classical Machine Learning model (Logistic Regression).

```http
POST /predict-ml
Content-Type: application/json
```

#### Request Body

```json
{
  "text": "This product is absolutely amazing! I love it."
}
```

| Field  | Type   | Required | Description     |
| ------ | ------ | -------- | --------------- |
| `text` | string | Yes      | Text to analyze |

#### Response

```json
{
  "prediction": "positive",
  "inference_time_ms": 0.45,
  "model": "LogisticRegression"
}
```

| Field               | Type   | Description               |
| ------------------- | ------ | ------------------------- |
| `prediction`        | string | "positive" or "negative"  |
| `inference_time_ms` | float  | Time taken for inference  |
| `model`             | string | Model used for prediction |

#### Example

**cURL:**

```bash
curl -X POST http://localhost:8000/predict-ml \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing! I love it."}'
```

**Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict-ml",
    json={"text": "This product is absolutely amazing! I love it."}
)
print(response.json())
# {'prediction': 'positive', 'inference_time_ms': 0.45, 'model': 'LogisticRegression'}
```

**JavaScript:**

```javascript
const response = await fetch("http://localhost:8000/predict-ml", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "This product is absolutely amazing! I love it.",
  }),
});
const data = await response.json();
console.log(data);
// {prediction: 'positive', inference_time_ms: 0.45, model: 'LogisticRegression'}
```

---

### Predict with DL Model

Analyze sentiment using the Deep Learning model (DistilBERT).

```http
POST /predict-dl
Content-Type: application/json
```

#### Request Body

```json
{
  "text": "The movie was terrible, I wasted two hours of my life."
}
```

| Field  | Type   | Required | Description     |
| ------ | ------ | -------- | --------------- |
| `text` | string | Yes      | Text to analyze |

#### Response

```json
{
  "prediction": "negative",
  "inference_time_ms": 15.23,
  "model": "DistilBERT"
}
```

| Field               | Type   | Description               |
| ------------------- | ------ | ------------------------- |
| `prediction`        | string | "positive" or "negative"  |
| `inference_time_ms` | float  | Time taken for inference  |
| `model`             | string | Model used for prediction |

#### Example

**cURL:**

```bash
curl -X POST http://localhost:8000/predict-dl \
  -H "Content-Type: application/json" \
  -d '{"text": "The movie was terrible, I wasted two hours of my life."}'
```

**Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict-dl",
    json={"text": "The movie was terrible, I wasted two hours of my life."}
)
print(response.json())
# {'prediction': 'negative', 'inference_time_ms': 15.23, 'model': 'DistilBERT'}
```

---

## Error Handling

### Model Not Available

If a model hasn't been loaded (missing model file), you'll receive:

```json
{
  "detail": "ML Model not available"
}
```

**HTTP Status**: 503 Service Unavailable

### Invalid Request

If the request body is malformed:

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**HTTP Status**: 422 Unprocessable Entity

---

## Batch Processing Example

For processing multiple texts, loop through your data:

```python
import requests
from concurrent.futures import ThreadPoolExecutor

texts = [
    "Great product, highly recommend!",
    "Terrible experience, never again.",
    "It was okay, nothing special.",
    "Absolutely fantastic service!",
    "Waste of money."
]

def predict(text):
    response = requests.post(
        "http://localhost:8000/predict-ml",
        json={"text": text}
    )
    return response.json()

# Sequential processing
results = [predict(text) for text in texts]

# Parallel processing (faster)
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(predict, texts))

for text, result in zip(texts, results):
    print(f"{text[:30]}... -> {result['prediction']}")
```

---

## OpenAPI Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide an interactive interface to test endpoints directly in your browser.

---

## Rate Limiting

The current implementation does not include rate limiting. For production, consider:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict-ml")
@limiter.limit("100/minute")
def predict_ml(request: Request, body: PredictionRequest):
    # ...
```

---

## Response Times

Expected response times under normal load:

| Endpoint       | Typical Latency | 99th Percentile |
| -------------- | --------------- | --------------- |
| `/healthcheck` | <5ms            | <10ms           |
| `/predict-ml`  | <1ms            | <5ms            |
| `/predict-dl`  | ~15ms           | ~50ms           |

---

## SDK Examples

### Python Client Class

```python
import requests
from typing import Literal

class SentimentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def healthcheck(self) -> dict:
        response = requests.get(f"{self.base_url}/healthcheck")
        return response.json()

    def predict(
        self,
        text: str,
        model: Literal["ml", "dl"] = "ml"
    ) -> dict:
        endpoint = f"/predict-{model}"
        response = requests.post(
            f"{self.base_url}{endpoint}",
            json={"text": text}
        )
        return response.json()

# Usage
client = SentimentClient()
print(client.healthcheck())
print(client.predict("This is great!", model="ml"))
print(client.predict("This is terrible!", model="dl"))
```

### JavaScript/TypeScript Client

```typescript
class SentimentClient {
  constructor(private baseUrl: string = "http://localhost:8000") {}

  async healthcheck(): Promise<HealthCheckResponse> {
    const response = await fetch(`${this.baseUrl}/healthcheck`);
    return response.json();
  }

  async predict(
    text: string,
    model: "ml" | "dl" = "ml",
  ): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseUrl}/predict-${model}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    return response.json();
  }
}

// Usage
const client = new SentimentClient();
const health = await client.healthcheck();
const prediction = await client.predict("This is amazing!", "ml");
```
