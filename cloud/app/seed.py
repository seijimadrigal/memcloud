"""Seed script — create default org + API key."""
import asyncio
import hashlib
import secrets
import sys

from sqlalchemy import select
from app.database import engine, async_session, init_db
from app.models import Organization, ApiKey


async def seed():
    await init_db()

    async with async_session() as db:
        # Check if already seeded
        result = await db.execute(select(Organization).limit(1))
        if result.scalar_one_or_none():
            print("Already seeded.")
            # Print existing key prefix
            keys = (await db.execute(select(ApiKey))).scalars().all()
            for k in keys:
                print(f"  Key: {k.key_prefix}... (org: {k.org_id})")
            return

        # Create default org
        org = Organization(id="default", name="Default Organization")
        db.add(org)
        await db.flush()

        # Generate API key
        raw_key = f"mc_{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = ApiKey(
            org_id="default",
            key_hash=key_hash,
            key_prefix=raw_key[:10],
            name="Default Key",
            permissions={"read": ["*"], "write": ["*"]},
        )
        db.add(api_key)
        await db.commit()

        print(f"✅ Seeded default org")
        print(f"🔑 API Key: {raw_key}")
        print(f"   Save this! It won't be shown again.")


if __name__ == "__main__":
    asyncio.run(seed())
