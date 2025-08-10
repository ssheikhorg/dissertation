import httpx
from typing import List
from pydantic import BaseModel
from app.config import settings
import time


class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    publication_date: str
    journal: str


class PubMedRetriever:
    def __init__(self, api_key: str = None):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.api_key = api_key or settings.pubmed_api_key
        self.cache = {}  # Simple cache to avoid duplicate requests
        self.last_request_time = 0
        self.min_request_interval = (
            0.34  # Respect PubMed's rate limit (3 requests/second)
        )

    def _rate_limit(self):
        """Ensure we respect PubMed's rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    async def search(self, query: str, max_results: int = 3) -> List[PubMedArticle]:
        """Search PubMed for relevant articles"""
        cache_key = f"search:{query}:{max_results}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._rate_limit()
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with httpx.AsyncClient() as client:
            search_url = f"{self.base_url}/esearch.fcgi"
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

            article_ids = data.get("esearchresult", {}).get("idlist", [])
            if not article_ids:
                return []

            articles = await self.fetch_articles(article_ids)
            self.cache[cache_key] = articles
            return articles

    async def fetch_articles(self, pmid_list: List[str]) -> List[PubMedArticle]:
        """Fetch full article details by PMID"""
        cache_key = f"articles:{','.join(pmid_list)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._rate_limit()
        params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key

        async with httpx.AsyncClient() as client:
            fetch_url = f"{self.base_url}/efetch.fcgi"
            response = await client.get(fetch_url, params=params)
            response.raise_for_status()

            # Parse XML response (simplified - in practice you'd use lxml or similar)
            articles = self._parse_pubmed_xml(response.text)
            self.cache[cache_key] = articles
            return articles

    def _parse_pubmed_xml(self, xml_content: str) -> List[PubMedArticle]:
        """Simplified XML parser for PubMed results"""
        # In a real implementation, you'd use lxml or similar for proper parsing
        # This is a simplified version that would need proper XML parsing
        articles = []
        # Example parsing logic (would need proper XML parsing):
        # Split by <PubmedArticle> tags and extract relevant fields
        return articles

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant medical context for a query"""
        articles = await self.search(query, max_results=top_k)
        if not articles:
            return "No relevant medical context found."

        context = []
        for article in articles:
            context.append(f"Title: {article.title}")
            if article.abstract:
                context.append(f"Abstract: {article.abstract}")
            context.append(f"Journal: {article.journal}, {article.publication_date}")
            context.append("---")

        return "\n".join(context)[:5000]  # Limit context length
