# Age Estimation API

Single-file FastAPI application that estimates age from an uploaded image using the Qwen2.5-VL-3B-Instruct model.
All application logic is contained in `app.py`. The application is started using the provided `run.sh` shell script.

## Setup & Run
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```
```bash
./run.sh
```

## Endpoints

### GET /health
Returns the health status and system information.

### POST /estimate-age
Accepts an image file (JPG, JPEG, or PNG) and returns an estimated age along with other metadata.

## Example Usage

```bash
curl --location 'http://127.0.0.1:8000/estimate-age' \
--form 'image=@"/Users/chrislacaille/Documents/ai-learning/qwen-api/img/mid-30s.jpg"'
```
