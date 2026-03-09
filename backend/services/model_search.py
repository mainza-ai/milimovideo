"""
Model Search Service — Live HuggingFace Hub discovery.

Wraps HfApi.list_models() with curated filters for LTX-2 and Flux model families.
Results can be cross-referenced with the local manifest to show download status.

Usage:
    from services.model_search import model_search
    results = await model_search.search("camera control", family="ltx2")
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Curated author organizations for supported models
CURATED_AUTHORS = {
    "ltx2": ["Lightricks"],
    "ltx23": ["Lightricks"],
    "flux2": ["black-forest-labs"],
    "sam3": ["facebook"],
}

# Pipeline tag mappings for filtering
FAMILY_PIPELINE_TAGS = {
    "ltx2": ["image-to-video", "text-to-video"],
    "ltx23": ["image-to-video", "text-to-video"],
    "flux2": ["text-to-image"],
    "sam3": ["image-segmentation"],
}


class ModelSearchService:
    """
    Searches HuggingFace Hub for compatible models.
    
    Combines live Hub results with local manifest status
    to show which discovered models are already downloaded.
    """

    def __init__(self):
        self._api = None

    def _get_api(self):
        """Lazy init HfApi to avoid import on startup."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi
                self._api = HfApi()
            except ImportError:
                raise ImportError(
                    "huggingface_hub required for model search. "
                    "Install with: pip install huggingface_hub"
                )
        return self._api

    async def search(
        self,
        query: str = "",
        family: Optional[str] = None,
        limit: int = 25,
    ) -> list[dict]:
        """
        Search HuggingFace Hub for models matching the query.
        
        Args:
            query: Free-text search term (e.g. "camera control", "IC-LoRA")
            family: Filter to a specific family (ltx2, flux2, sam3)
            limit: Max results per author
            
        Returns:
            List of model dicts with Hub metadata + local status.
        """
        import asyncio

        api = self._get_api()

        # Determine which authors to search
        if family and family in CURATED_AUTHORS:
            authors = CURATED_AUTHORS[family]
        else:
            authors = []
            for author_list in CURATED_AUTHORS.values():
                authors.extend(author_list)
            authors = list(set(authors))

        results = []

        for author in authors:
            try:
                # Run the blocking HfApi call in a thread
                models = await asyncio.to_thread(
                    api.list_models,
                    author=author,
                    search=query if query else None,
                    sort="downloads",
                    direction=-1,
                    limit=limit,
                )

                for m in models:
                    result = self._format_model(m)
                    if result:
                        results.append(result)

            except Exception as e:
                logger.warning(f"Hub search failed for author '{author}': {e}")

        # Cross-reference with local registry
        self._annotate_local_status(results)

        # Sort by downloads (descending)
        results.sort(key=lambda r: r.get("downloads", 0), reverse=True)

        return results

    async def get_model_info(self, repo_id: str) -> Optional[dict]:
        """
        Get detailed info for a specific HuggingFace repo.
        
        Includes file listing with sizes for download planning.
        """
        import asyncio

        api = self._get_api()

        try:
            info = await asyncio.to_thread(
                api.model_info,
                repo_id,
                files_metadata=True,
            )

            files = []
            total_size = 0
            for f in (info.siblings or []):
                file_info = {
                    "filename": f.rfilename,
                    "size_bytes": getattr(f, "size", None),
                }
                files.append(file_info)
                if file_info["size_bytes"]:
                    total_size += file_info["size_bytes"]

            return {
                "repo_id": info.id,
                "author": info.author,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags or [],
                "pipeline_tag": info.pipeline_tag,
                "gated": info.gated if info.gated else False,
                "last_modified": str(info.last_modified) if info.last_modified else None,
                "files": files,
                "total_size_bytes": total_size,
                "card_data": info.card_data.to_dict() if info.card_data else None,
            }

        except Exception as e:
            logger.error(f"Failed to get model info for '{repo_id}': {e}")
            return None

    def _format_model(self, model) -> Optional[dict]:
        """Format a HfApi model result into a standardized dict."""
        try:
            return {
                "repo_id": model.id,
                "name": model.id.split("/")[-1],
                "author": model.author,
                "downloads": getattr(model, "downloads", 0),
                "likes": getattr(model, "likes", 0),
                "pipeline_tag": getattr(model, "pipeline_tag", None),
                "tags": getattr(model, "tags", []) or [],
                "last_modified": str(model.last_modified) if hasattr(model, "last_modified") and model.last_modified else None,
                "gated": getattr(model, "gated", False),
                "local_status": None,  # Will be filled by _annotate_local_status
                "in_manifest": False,
            }
        except Exception as e:
            logger.debug(f"Failed to format model: {e}")
            return None

    def _annotate_local_status(self, results: list[dict]) -> None:
        """Cross-reference Hub results with local manifest."""
        try:
            from services.model_registry import model_registry

            # Build a lookup from HF repo to local model
            repo_to_model = {}
            for model_dict in model_registry.get_all_models():
                hf = model_dict.get("huggingface", {})
                repo = hf.get("repo")
                if repo:
                    repo_to_model[repo] = model_dict

            for result in results:
                local = repo_to_model.get(result["repo_id"])
                if local:
                    result["local_status"] = local["status"]
                    result["in_manifest"] = True
                    result["local_model_id"] = local["id"]

        except Exception as e:
            logger.debug(f"Could not annotate local status: {e}")


# ── Singleton ──────────────────────────────────────────────────────
model_search = ModelSearchService()
