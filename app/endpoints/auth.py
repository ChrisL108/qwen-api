from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import API_TOKEN

security = HTTPBearer()

async def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verify the Bearer token provided in the Authorization header.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication credentials")
    
    token = credentials.credentials  # Extract the token from the Bearer auth
    
    # Check if token matches your configured API token
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return token