import logging
import re
from urllib.parse import urljoin
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

AI_JOB_KEYWORDS = {
    "machine learning", "ml engineer", "ai engineer", "llm", "nlp",
    "deep learning", "data scientist", "research scientist",
    "applied scientist", "foundation model", "computer vision",
    "reinforcement learning", "mlops", "ai infrastructure",
    "generative ai", "prompt engineer",
}

KNOWN_AI_COMPANY_BOARDS = [
    {"platform": "greenhouse", "company": "anthropic",    "url": "https://boards.greenhouse.io/anthropic"},
    {"platform": "greenhouse", "company": "cohere",       "url": "https://boards.greenhouse.io/cohere"},
    {"platform": "greenhouse", "company": "mistralai",    "url": "https://boards.greenhouse.io/mistralai"},
    {"platform": "greenhouse", "company": "huggingface",  "url": "https://boards.greenhouse.io/huggingface"},
    {"platform": "greenhouse", "company": "openai",       "url": "https://boards.greenhouse.io/openai"},
    {"platform": "lever",      "company": "scale-ai",     "url": "https://jobs.lever.co/scaleai"},
    {"platform": "lever",      "company": "together-ai",  "url": "https://jobs.lever.co/togetherai"},
    {"platform": "lever",      "company": "perplexity",   "url": "https://jobs.lever.co/perplexity"},
]


class JobScraper(BaseScraper):

    def __init__(self):
        super().__init__(source_name="JobBoards", base_url="https://boards.greenhouse.io")
        self._company_boards = KNOWN_AI_COMPANY_BOARDS

    def fetch_article_list(self, max_articles: int) -> list:
        job_urls = []
        seen = set()
        for board in self._company_boards:
            if len(job_urls) >= max_articles:
                break
            urls = self._get_board_job_urls(board)
            for url in urls:
                if url not in seen:
                    job_urls.append(url)
                    seen.add(url)
        logger.info(f"[JobScraper] Found {len(job_urls)} job posting URLs")
        return job_urls[:max_articles]

    def parse_article(self, html: str, url: str) -> dict:
        soup = self._make_soup(html)
        title = self._extract_text(soup, [
            "h1.app-title",
            "h1[class*='posting-headline']",
            "h1[data-qa='posting-name']",
            "h1",
        ])
        company = self._extract_company_from_url(url)
        body_el = (
            soup.select_one("div#content") or
            soup.select_one("div[class*='posting-description']") or
            soup.select_one("div[data-qa='posting-description']") or
            soup.select_one("section[class*='job-description']")
        )
        description = body_el.get_text(separator=" ", strip=True) if body_el else ""
        location = self._extract_text(soup, [
            "span.location",
            "span[data-qa='posting-location']",
            "div[class*='location']",
        ])
        skills = self._extract_skills(description)
        if not title or not description:
            return {}
        if not self._is_ai_job(title, description):
            return {}
        raw_text = (
            f"Job Title: {title}\n"
            f"Company: {company}\n"
            f"Location: {location}\n\n"
            f"{description}"
        )
        return {
            "title": f"{company} - {title}",
            "raw_text": raw_text,
            "published_date": None,
            "author": None,
            "tags": skills[:15],
            "metadata": {
                "job_title": title,
                "company": company,
                "location": location,
                "skills": skills,
                "document_type": "job_posting",
            },
        }

    def _get_board_job_urls(self, board: dict) -> list:
        html = self._get_with_retry(board["url"])
        if not html:
            return []
        soup = self._make_soup(html)
        urls = []
        if board["platform"] == "greenhouse":
            for a in soup.select("a.posting-title, a[href*='/jobs/']"):
                href = a.get("href", "")
                if href:
                    full_url = href if href.startswith("http") else urljoin(board["url"], href)
                    urls.append(full_url)
        elif board["platform"] == "lever":
            for a in soup.select("a[href*='jobs.lever.co']"):
                href = a.get("href", "")
                if href and "/apply" not in href:
                    urls.append(href)
        return urls

    def _is_ai_job(self, title: str, description: str) -> bool:
        text = (title + " " + description[:500]).lower()
        return any(kw in text for kw in AI_JOB_KEYWORDS)

    def _extract_skills(self, text: str) -> list:
        tech_skills = [
            "Python", "PyTorch", "TensorFlow", "JAX", "CUDA",
            "LLM", "Transformer", "BERT", "GPT", "Llama",
            "RAG", "Vector Database", "Pinecone", "Weaviate", "Chroma",
            "RLHF", "Fine-tuning", "Prompt Engineering",
            "MLOps", "Kubernetes", "Docker", "AWS", "GCP", "Azure",
            "Spark", "Ray", "NLP", "Computer Vision", "Reinforcement Learning",
        ]
        found = []
        text_lower = text.lower()
        for skill in tech_skills:
            if skill.lower() in text_lower:
                found.append(skill)
        return found

    def _extract_company_from_url(self, url: str) -> str:
        match = re.search(r"greenhouse\.io/([^/]+)", url)
        if match:
            return match.group(1).replace("-", " ").title()
        match = re.search(r"lever\.co/([^/]+)", url)
        if match:
            return match.group(1).replace("-", " ").title()
        return "Unknown"
