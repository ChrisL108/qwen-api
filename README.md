# Qwen-2.5-VL Model Gateway (OpenAI and Ollama Compatible)

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

The API will be available at https://coder1-demo.lively-video-dev-hoodoo.livelyvideo.tv.


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


## Usage Examples

### Python Client (OpenAI SDK)

```python
import openai

# Initialize the client with the local endpoint
client = openai.OpenAI(
    api_key="super-secret-key",
    base_url="https://coder1-demo.lively-video-dev-hoodoo.livelyvideo.tv/v1"
)

# Create the request
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[
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
                        "url": "base64-encoded-image-string <-OR-> url-to-image"
                    }
                }
            ]
        }
    ],
)

# Print the result
print(f"Estimated age: {response.choices[0].message.content}")
```

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
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
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

response = requests.post("https://coder1-demo.lively-video-dev-hoodoo.livelyvideo.tv/v1/chat/completions", json=payload)
print(f"Estimated age: {response.json()['choices'][0]['message']['content']}")
```

### Next.js Client (OpenAI SDK)

```python
'use client'

import OpenAI from 'openai'

const openai = new OpenAI({
	apiKey: 'super-secret-key',
	baseURL: 'https://coder1-demo.lively-video-dev-hoodoo.livelyvideo.tv/v1',
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
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "prompt": "What is the age of this person?",
    "images": [f"data:image/jpeg;base64,{base64_image}"],
    "stream": False
}

response = requests.post("https://coder1-demo.lively-video-dev-hoodoo.livelyvideo.tv/api/generate", json=payload)
print(f"Estimated age: {response.json()['response']}")
```

### cURL Example (OpenAI format)

```bash
curl -X POST https://coder1-demo.lively-video-dev-hoodoo.livelyvideo.tv/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
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
curl -X POST https://coder1-demo.lively-video-dev-hoodoo.livelyvideo.tv/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "prompt": "What is the age of this person?",
    "images": ["data:image/jpeg;base64,YOUR_BASE64_STRING_HERE"]
  }'
```

## Deployment

### Docker Deployment

Build and run the Docker container:

```bash
docker build -t qwen-2.5-vl-model-gateway .
docker run -p 7860:7860 qwen-2.5-vl-model-gateway
```


## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce maximum image resolution
   - Use a smaller model if available
   - Adjust `torch_dtype` for reduced precision

2. **Slow Processing**
   - Enable GPU acceleration by updating `device_map`
   - Use smaller images
   - Consider adding a caching layer


## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

