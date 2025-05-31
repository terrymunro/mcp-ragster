import logging
from typing import Any

import httpx

from ragster.config import settings
from ragster.exceptions import APICallError, PerplexityAPIError


logger = logging.getLogger(__name__)


class PerplexityAPIClient:
    """Client for Perplexity AI API."""

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        logger.info("Initializing PerplexityAPIClient.")
        self.http_client = http_client

    async def _make_request(self, payload: dict[str, Any]) -> str:
        """Make a request to the Perplexity API."""
        headers = {
            "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            if self.http_client:
                response = await self.http_client.post(
                    settings.PERPLEXITY_CHAT_API_URL, json=payload, headers=headers
                )
            else:
                async with httpx.AsyncClient(
                    timeout=settings.HTTP_TIMEOUT_PERPLEXITY
                ) as client:
                    response = await client.post(
                        settings.PERPLEXITY_CHAT_API_URL, json=payload, headers=headers
                    )

            if response.status_code != 200:
                raise APICallError(
                    "Perplexity", response.status_code, response.text[:500]
                )

            response_data = response.json()
            choices = response_data.get("choices")
            if (
                choices
                and isinstance(choices, list)
                and len(choices) > 0
                and choices[0].get("message")
            ):
                content = choices[0]["message"].get("content")
                if content and isinstance(content, str):
                    return content.strip()

            raise PerplexityAPIError(
                f"Perplexity response format unexpected or empty content: {str(response_data)[:200]}"
            )
        except httpx.HTTPStatusError as e:
            raise APICallError(
                "Perplexity", e.response.status_code, e.response.text[:200]
            ) from e
        except httpx.RequestError as e:
            raise PerplexityAPIError(
                f"Network request to Perplexity failed: {e}", underlying_error=e
            )
        except Exception as e:
            if isinstance(e, (PerplexityAPIError, APICallError)):
                raise
            raise PerplexityAPIError(
                f"Unexpected error querying Perplexity: {e}", underlying_error=e
            )

    async def check_fact(self, fact: str) -> str:
        """Fact-check content using Perplexity AI."""
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": """
You are a professional fact-checker with extensive research capabilities. Your task is to evaluate claims or articles for factual accuracy. Focus on identifying false, misleading, or unsubstantiated claims.

## Evaluation Process
For each piece of content, you will:
1. Identify specific claims that can be verified
2. Research each claim thoroughly using the most reliable sources available
3. Determine if each claim is:
   - TRUE: Factually accurate and supported by credible evidence
   - FALSE: Contradicted by credible evidence
   - MISLEADING: Contains some truth but presents information in a way that could lead to incorrect conclusions
   - UNVERIFIABLE: Cannot be conclusively verified with available information
4. For claims rated as FALSE or MISLEADING, explain why and provide corrections

## Rating Criteria
- TRUE: Claim is supported by multiple credible sources with no significant contradicting evidence
- FALSE: Claim is contradicted by clear evidence from credible sources
- MISLEADING: Claim contains factual elements but is presented in a way that omits crucial context or leads to incorrect conclusions
- UNVERIFIABLE: Claim cannot be conclusively verified or refuted with available evidence

## Guidelines
- Remain politically neutral and focus solely on factual accuracy
- Do not use political leaning as a factor in your evaluation
- Prioritize official data, peer-reviewed research, and reports from credible institutions
- Cite specific, reliable sources for your determinations
- Consider the context and intended meaning of statements
- Distinguish between factual claims and opinions
- Pay attention to dates, numbers, and specific details
- Be precise and thorough in your explanations

## Response Format
Respond in JSON format with the following structure:
```json
{
    "overall_rating": "MOSTLY_TRUE|MIXED|MOSTLY_FALSE",
    "summary": "Brief summary of your overall findings",
    "claims": [
        {
            "claim": "The specific claim extracted from the text",
            "rating": "TRUE|FALSE|MISLEADING|UNVERIFIABLE",
            "explanation": "Your explanation with supporting evidence",
            "sources": ["Source 1", "Source 2"]
        },
        // Additional claims...
    ]
}
```

## Criteria for Overall Rating
- MOSTLY_TRUE: Most claims are true, with minor inaccuracies that don't affect the main message
- MIXED: The content contains a roughly equal mix of true and false/misleading claims
- MOSTLY_FALSE: Most claims are false or misleading, significantly distorting the facts

Ensure your evaluation is thorough, fair, and focused solely on factual accuracy. Do not allow personal bias to influence your assessment. Be especially rigorous with claims that sound implausible or extraordinary.
""",
                },
                {
                    "role": "user",
                    "content": f"Fact check the following text and identify any false or misleading claims:\n\n{fact}",
                },
            ],
        }
        return await self._make_request(payload)

    async def query(self, topic: str) -> str:
        """Research a topic using Perplexity AI."""
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert research assistant with access to real-time web search capabilities. Your task is to conduct comprehensive, in-depth research on any given topic.

## Research Methodology
For each topic, you will:
1. **Comprehensive Coverage**: Research all major aspects, subtopics, and dimensions of the subject
2. **Current Information**: Prioritize recent developments, trends, and up-to-date information
3. **Multiple Perspectives**: Include different viewpoints, approaches, and schools of thought
4. **Authoritative Sources**: Cite credible sources, academic papers, official reports, and expert opinions
5. **Context & Background**: Provide historical context and foundational knowledge
6. **Practical Applications**: Include real-world applications, use cases, and examples
7. **Future Outlook**: Discuss trends, predictions, and emerging developments

## Research Structure
Organize your research with clear sections:
- **Overview**: Brief introduction and key concepts
- **Current State**: Latest developments and current understanding
- **Key Players/Organizations**: Important figures, companies, institutions
- **Technical Details**: In-depth explanations of mechanisms, processes, or methodologies
- **Applications & Use Cases**: Real-world implementations and examples
- **Challenges & Limitations**: Known issues, controversies, or obstacles
- **Future Directions**: Emerging trends and potential developments
- **Sources**: Key references and citations

## Research Quality Standards
- Use multiple credible sources for each major claim
- Include quantitative data, statistics, and metrics where available
- Distinguish between established facts and emerging theories
- Note any conflicting information or debates in the field
- Provide specific examples and case studies
- Include relevant technical specifications or parameters

## Output Format
Present findings in a well-structured, comprehensive report that balances depth with readability. Use clear headings, bullet points, and logical flow. Aim for thoroughness while maintaining clarity.""",
                },
                {
                    "role": "user",
                    "content": f"Conduct comprehensive research on: {topic}",
                },
            ],
            "max_tokens": 4000,
            "return_related_questions": True,
            "search_recency_filter": "year",
            "search_domain_filter": ["-reddit.com", "-quora.com"],
        }
        return await self._make_request(payload)

    async def close(self):
        """Cleanup method for consistency."""
        pass
