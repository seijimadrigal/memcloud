"""API key authentication and rate limiting."""
import hashlib
import time
from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.database import get_db
from app.models import ApiKey, Organization
from app.config import REDIS_URL

security = HTTPBearer()
_redis = None


async def get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class AuthContext:
    def __init__(self, api_key: ApiKey, org: Organization):
        self.api_key = api_key
        self.org = org
        self.org_id = org.id
        self.permissions = api_key.permissions or {}


async def authenticate(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_db),
) -> AuthContext:
    token = credentials.credentials
    key_h = hash_key(token)

    result = await db.execute(select(ApiKey).where(ApiKey.key_hash == key_h, ApiKey.is_active == True))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Rate limiting
    r = await get_redis()
    now_min = int(time.time() // 60)
    now_day = int(time.time() // 86400)
    
    min_key = f"rl:{api_key.id}:{now_min}"
    day_key = f"rl:{api_key.id}:d:{now_day}"

    pipe = r.pipeline()
    pipe.incr(min_key)
    pipe.expire(min_key, 120)
    pipe.incr(day_key)
    pipe.expire(day_key, 172800)
    results = await pipe.execute()

    if results[0] > api_key.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (per minute)")
    if results[2] > api_key.rate_limit_per_day:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (per day)")

    org_result = await db.execute(select(Organization).where(Organization.id == api_key.org_id))
    org = org_result.scalar_one()

    return AuthContext(api_key=api_key, org=org)


def check_permission(api_key_record, required: str) -> bool:
    """Check if API key has required permission (read/write/admin)."""
    perms = api_key_record.permissions or ["*"]
    return "*" in perms or required in perms
