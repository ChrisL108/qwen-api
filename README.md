# Age Estimation API

A dual-compatible API for age estimation using Qwen2.5-VL vision language model.

- **OpenAI Compatible:** Use the standard OpenAI chat completions format
- **Ollama Compatible:** Use the simpler Ollama generation format


## Setup & Run
#### Create and activate virtual environment
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```
#### Install dependencies
```bash
pip install -r requirements.txt
```
#### Run the API
```bash
./run.sh
```

The API will be available at http://localhost:7860.


## API Documentation

The API supports two interface styles: OpenAI and Ollama.

- **All endpoints that accept images accept a base64 encoded image string or a URL to an image.**

### Health Check Endpoints

#### Universal Health Check

```
GET /health
```

Returns the API's current health status.

#### Ollama-style Health Check

```
GET /api/health
```

Returns the API's current health status in Ollama-compatible format.

### OpenAI-Compatible Endpoints

#### List Models

```
GET /v1/models
```

Returns a list of available models in OpenAI format.

#### Create Chat Completion

```
POST /v1/chat/completions
```

Request body (`image_url.url` can also be a URL to an image):

```json
{
  "model": "age-estimation",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is the age of this person? Please reply with just the number."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,YOUR_BASE64_STRING_HERE"
          }
        }
      ]
    }
  ],
  "stream": false
}
```

Response:

```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1709123456,
  "model": "age-estimation",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "35"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 34,
    "completion_tokens": 2,
    "total_tokens": 36
  }
}
```

### Ollama-Compatible Endpoints

#### List Models

```
GET /api/tags
```

Returns a list of available models in Ollama format.

#### Generate

```
POST /api/generate
```

Request body:

```json
{
  "model": "age-estimation",
  "prompt": "What is the age of this person?",
  "images": ["data:image/jpeg;base64,YOUR_BASE64_STRING_HERE"],
  "stream": false,
  "options": {}
}
```

Response:

```json
{
  "model": "age-estimation",
  "created_at": "2023-01-01T12:00:00Z",
  "response": "35",
  "done": true,
  "total_duration": 450000000
}
```

## Streaming Responses

Both API formats support streaming responses by setting `"stream": true` in the request. The responses will be sent incrementally as they become available.

### OpenAI streaming format

```
data: {"id":"chatcmpl-1234567890","object":"chat.completion.chunk","created":1709123456,"model":"age-estimation","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-1234567890","object":"chat.completion.chunk","created":1709123456,"model":"age-estimation","choices":[{"index":0,"delta":{"content":"35"},"finish_reason":null}]}

data: {"id":"chatcmpl-1234567890","object":"chat.completion.chunk","created":1709123456,"model":"age-estimation","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Ollama streaming format

```
{"model":"age-estimation","created_at":"2023-01-01T12:00:00Z","response":"35","done":false}
{"model":"age-estimation","created_at":"2023-01-01T12:00:00Z","response":"35","done":true,"total_duration":450000000}
```

## Usage Examples

### Python Client (OpenAI Format)

```python
import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode image
base64_image = encode_image("path/to/your/image.jpg")

# OpenAI format
payload = {
    "model": "age-estimation",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the age of this person?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
}

response = requests.post("http://localhost:7860/v1/chat/completions", json=payload)
print(f"Estimated age: {response.json()['choices'][0]['message']['content']}")
```

### Next.js Client (OpenAI Format)

```python
'use client'

import OpenAI from 'openai'

const openai = new OpenAI({
	apiKey: 'not-needed',
	baseURL: 'http://127.0.0.1:7860/v1',
	dangerouslyAllowBrowser: true 
})

const response = await openai.chat.completions.create({
	model: 'age-estimation',
	messages: [
		{
			role: 'user',
			content: [
				{
					type: 'text',
					text: 'What is the age of this person? Please reply with just the number.'
				},
				{
					type: 'image_url',
					image_url: {
						url: 'base64-encoded-image-string <-OR-> url-to-image'
					}
				}
			]
		}
	],
	max_tokens: 300,
})

console.log(`Estimated age: ${response.choices[0].message.content}`)
```

### Python Client (Ollama Format)

```python
import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode image
base64_image = encode_image("path/to/your/image.jpg")

# Ollama format
payload = {
    "model": "age-estimation",
    "prompt": "What is the age of this person?",
    "images": [f"data:image/jpeg;base64,{base64_image}"],
    "stream": False
}

response = requests.post("http://localhost:7860/api/generate", json=payload)
print(f"Estimated age: {response.json()['response']}")
```

### cURL Example (OpenAI format)

```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "age-estimation",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is the age of this person?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,YOUR_BASE64_STRING_HERE"
            }
          }
        ]
      }
    ]
  }'
```

### cURL Example (Ollama format)

```bash
curl -X POST http://localhost:7860/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "age-estimation",
    "prompt": "What is the age of this person?",
    "images": ["data:image/jpeg;base64,YOUR_BASE64_STRING_HERE"]
  }'
```

## Deployment

### Docker Deployment

Build and run the Docker container:

```bash
docker build -t age-estimation-api .
docker run -p 7860:7860 age-estimation-api
```

### Production Considerations

For production deployments:

1. Use a reverse proxy like Nginx for SSL termination and load balancing
2. Set `device_map` and `torch_dtype` according to your hardware capabilities
3. Consider using a process manager like Supervisor or systemd
4. Use a production-grade ASGI server like Gunicorn with Uvicorn workers

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check your Hugging Face token is set
   - Ensure enough disk space for model downloads
   - Verify network connectivity to Hugging Face servers

2. **Memory Issues**
   - Reduce maximum image resolution
   - Use a smaller model if available
   - Adjust `torch_dtype` for reduced precision

3. **Slow Processing**
   - Enable GPU acceleration by updating `device_map`
   - Use smaller images
   - Consider adding a caching layer

### Logs

The API uses Python's logging module with INFO level by default. Check logs for details about any issues.

## License

[Include your license information here]

