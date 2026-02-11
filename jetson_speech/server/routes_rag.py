"""
RAG (Retrieval Augmented Generation) routes for the speech server.

Provides endpoints for managing RAG collections and searching.
"""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from jetson_speech.server.schemas import (
    RAGIngestRequest,
    RAGIngestResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGSearchResult,
    RAGInfoResponse,
    RAGCollectionsResponse,
)

router = APIRouter(prefix="/rag", tags=["rag"])

# Global RAG instances (keyed by collection name)
_rag_instances: dict = {}


def get_rag(collection: str, create_if_missing: bool = False):
    """Get or create RAG instance for a collection."""
    global _rag_instances

    if collection not in _rag_instances:
        try:
            from jetson_speech.rag import RAGPipeline
            rag = RAGPipeline(collection_name=collection)

            if rag.count() == 0 and not create_if_missing:
                return None  # Collection is empty and we don't want to create

            _rag_instances[collection] = rag
            print(f"RAG loaded: {collection} ({rag.count()} chunks)", file=sys.stderr)

        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"RAG dependencies not installed: {e}"
            )

    return _rag_instances.get(collection)


@router.get("/collections", response_model=RAGCollectionsResponse)
async def list_collections() -> RAGCollectionsResponse:
    """List all available RAG collections."""
    try:
        from jetson_speech.config import get_default_cache_dir
        import chromadb
        from chromadb.config import Settings

        persist_dir = get_default_cache_dir() / "rag"
        if not persist_dir.exists():
            return RAGCollectionsResponse(success=True, collections=[])

        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        collections = [c.name for c in client.list_collections()]
        return RAGCollectionsResponse(success=True, collections=collections)

    except Exception as e:
        return RAGCollectionsResponse(
            success=False,
            collections=[],
            error=str(e)
        )


@router.get("/{collection}/info", response_model=RAGInfoResponse)
async def get_collection_info(collection: str) -> RAGInfoResponse:
    """Get information about a RAG collection."""
    rag = get_rag(collection, create_if_missing=False)

    if rag is None:
        return RAGInfoResponse(
            success=False,
            collection=collection,
            chunk_count=0,
            sources=[],
            embedding_model="",
            error=f"Collection '{collection}' not found or is empty"
        )

    try:
        return RAGInfoResponse(
            success=True,
            collection=collection,
            chunk_count=rag.count(),
            sources=rag.get_sources(),
            embedding_model=rag.embedding_model.model_name,
        )
    except Exception as e:
        return RAGInfoResponse(
            success=False,
            collection=collection,
            chunk_count=0,
            sources=[],
            embedding_model="",
            error=str(e)
        )


@router.post("/{collection}/search", response_model=RAGSearchResponse)
async def search_collection(collection: str, request: RAGSearchRequest) -> RAGSearchResponse:
    """Search a RAG collection for relevant documents."""
    rag = get_rag(collection, create_if_missing=False)

    if rag is None:
        return RAGSearchResponse(
            success=False,
            results=[],
            query=request.query,
            error=f"Collection '{collection}' not found or is empty"
        )

    try:
        results = rag.retrieve(request.query, top_k=request.top_k)

        return RAGSearchResponse(
            success=True,
            results=[
                RAGSearchResult(
                    content=r["content"],
                    score=r["score"],
                    source=r.get("metadata", {}).get("source", "unknown"),
                    metadata=r.get("metadata", {}),
                )
                for r in results
            ],
            query=request.query,
        )
    except Exception as e:
        return RAGSearchResponse(
            success=False,
            results=[],
            query=request.query,
            error=str(e)
        )


@router.post("/{collection}/ingest", response_model=RAGIngestResponse)
async def ingest_documents(collection: str, request: RAGIngestRequest) -> RAGIngestResponse:
    """Ingest documents into a RAG collection."""
    # Override collection from path
    request.collection = collection

    rag = get_rag(collection, create_if_missing=True)

    if rag is None:
        return RAGIngestResponse(
            success=False,
            collection=collection,
            documents_added=0,
            chunks_added=0,
            error="Failed to create RAG collection"
        )

    try:
        if request.source_type == "url":
            count = rag.ingest_url(
                request.content,
                follow_links=request.follow_links,
                max_pages=request.max_pages,
            )
            return RAGIngestResponse(
                success=True,
                collection=collection,
                documents_added=1,
                chunks_added=count,
            )

        elif request.source_type == "text":
            count = rag.ingest_text(request.content, source="api_text")
            return RAGIngestResponse(
                success=True,
                collection=collection,
                documents_added=1,
                chunks_added=count,
            )

        elif request.source_type == "file":
            path = Path(request.content)
            if not path.exists():
                return RAGIngestResponse(
                    success=False,
                    collection=collection,
                    documents_added=0,
                    chunks_added=0,
                    error=f"File not found: {request.content}"
                )
            count = rag.ingest_file(path)
            return RAGIngestResponse(
                success=True,
                collection=collection,
                documents_added=1,
                chunks_added=count,
            )

        elif request.source_type == "directory":
            path = Path(request.content)
            if not path.exists():
                return RAGIngestResponse(
                    success=False,
                    collection=collection,
                    documents_added=0,
                    chunks_added=0,
                    error=f"Directory not found: {request.content}"
                )
            count = rag.ingest_directory(path, glob_pattern=request.glob_pattern)
            return RAGIngestResponse(
                success=True,
                collection=collection,
                documents_added=-1,  # Unknown for directory
                chunks_added=count,
            )

        else:
            return RAGIngestResponse(
                success=False,
                collection=collection,
                documents_added=0,
                chunks_added=0,
                error=f"Unknown source type: {request.source_type}"
            )

    except Exception as e:
        return RAGIngestResponse(
            success=False,
            collection=collection,
            documents_added=0,
            chunks_added=0,
            error=str(e)
        )


@router.delete("/{collection}")
async def delete_collection(collection: str):
    """Delete a RAG collection."""
    global _rag_instances

    rag = get_rag(collection, create_if_missing=False)

    if rag is None:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection}' not found"
        )

    try:
        rag.clear()
        # Remove from cache
        if collection in _rag_instances:
            del _rag_instances[collection]

        return {"success": True, "message": f"Collection '{collection}' deleted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{collection}/context")
async def build_context(collection: str, query: str, top_k: int = 3):
    """Build context string for a query (used by LLM endpoint internally)."""
    rag = get_rag(collection, create_if_missing=False)

    if rag is None:
        return {
            "success": False,
            "context": "",
            "error": f"Collection '{collection}' not found"
        }

    try:
        context = rag.build_context(query, top_k=top_k)
        return {
            "success": True,
            "context": context,
            "query": query,
        }
    except Exception as e:
        return {
            "success": False,
            "context": "",
            "error": str(e)
        }
