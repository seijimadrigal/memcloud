"""PostgreSQL database setup with async SQLAlchemy."""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import DATABASE_URL

engine = create_async_engine(DATABASE_URL, pool_size=20, max_overflow=10)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        yield session


async def init_db():
    from sqlalchemy import text as sa_text
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Use checkfirst=True to avoid race conditions with multiple workers
        await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=True))
        try:
            await conn.execute(sa_text("""
                CREATE INDEX IF NOT EXISTS idx_memories_fts
                ON memories USING gin(to_tsvector('english', content))
            """))
        except Exception:
            pass
