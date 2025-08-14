# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "codewords-client==0.3.0",
#   "fastapi==0.116.1",
#   "firecrawl==2.16.0",
#   "openai==1.99.7"
# ]
# [tool.env-checker]
# env_vars = [
#   "PORT=8000",
#   "LOGLEVEL=INFO",
#   "CODEWORDS_API_KEY",
#   "CODEWORDS_RUNTIME_URI",
#   "FIRECRAWL_API_KEY"
# ]
# ///

import asyncio
import os
from textwrap import dedent
from typing import Literal
from urllib.parse import urljoin, urlparse

from codewords_client import AsyncCodewordsClient, logger, run_service
from fastapi import FastAPI, HTTPException
from firecrawl import FirecrawlApp
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator

class ArticleAnalysis(BaseModel):
    article_type: Literal["news", "opinion", "editorial", "analysis"] = Field(description="Type of article content")
    factual_claims: list[str] = Field(description="List of specific factual claims that can be verified")
    bias_indicators: list[str] = Field(description="List of potential bias or manipulation indicators found")
    credibility_score: int = Field(description="Initial credibility assessment from 1-10", ge=1, le=10)
    
class VerificationResult(BaseModel):
    claim: str = Field(description="The original claim being verified")
    verification_status: Literal["supported", "disputed", "unclear", "no_evidence"] = Field(description="Verification outcome")
    supporting_sources: list[str] = Field(description="URLs of sources that support the claim")
    disputing_sources: list[str] = Field(description="URLs of sources that dispute the claim")
    evidence_summary: str = Field(description="Summary of evidence found")

async def _extract_article_content(url: str) -> dict:
    """Extract article content from URL using Firecrawl."""
    logger.info("Step 1/5: Extracting article content", url=url)

    # Validate URL format
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL format. Please provide a complete URL including http:// or https://")
    
    client = FirecrawlApp()
    
    scraped_data = await asyncio.to_thread(
        client.scrape_url,
        url,
        params={
            "formats": ["markdown"],
            "timeout": 15000,
            "waitFor": 2000
        }
    )
    
    if not scraped_data or not scraped_data.markdown:
        raise HTTPException(status_code=400, detail="Failed to extract content from the provided URL")
        
    content = {
        "markdown": scraped_data.markdown,
        "title": scraped_data.metadata.get("title") if scraped_data.metadata else "No title found",
        "description": scraped_data.metadata.get("description") if scraped_data.metadata else "",
        "source_domain": parsed.netloc
    }
    
    logger.debug("Successfully extracted article content", 
                url=url, 
                content_length=len(content["markdown"]), 
                title=content["title"])
    return content

async def _analyze_article_content(content: dict) -> ArticleAnalysis:
    """Analyze article content to classify type and extract claims."""
    logger.info("Step 2/5: Analyzing article content and extracting claims", title=content["title"])

    client = AsyncOpenAI(base_url=urljoin(os.environ["CODEWORDS_RUNTIME_URI"], "run/gemini/v1"))
    
    prompt = dedent("""\
        Analyze the following news article and classify its content type, extract factual claims, and identify potential bias indicators.
        
        <article>
        Title: {title}
        Content: {markdown}
        Source Domain: {source_domain}
        </article>

        Your analysis should:
        1. Determine if this is a news article, opinion piece, editorial, or analysis piece
        2. Extract specific factual claims that can be verified (avoid opinions)
        3. Identify potential bias indicators or manipulation techniques
        4. Provide an initial credibility assessment (1-10) based on writing quality and journalistic standards
        
        Focus on factual, verifiable claims only. Ignore opinions, speculation, and subjective statements.\
        """).format(
        title=content["title"],
        markdown=content["markdown"][:8000],  # Limit content to avoid token limits
        source_domain=content["source_domain"]
    )

    response = await client.beta.chat.completions.parse(
        model="gemini-2.5-pro",
        messages=[
            {"role": "system", "content": "You are an expert journalist and fact-checker who can identify different types of articles and extract verifiable claims."},
            {"role": "user", "content": prompt}
        ],
        response_format=ArticleAnalysis,
        temperature=0.2
    )
    
    analysis = response.choices[0].message.parsed
    logger.debug("Completed article analysis", 
                article_type=analysis.article_type,
                num_claims=len(analysis.factual_claims),
                credibility_score=analysis.credibility_score)
    return analysis

async def _verify_factual_claims(claims: list[str], article_title: str) -> list[VerificationResult]:
    """Verify factual claims against reliable sources using web search."""
    if not claims:
        logger.info("Step 3/5: No factual claims to verify")
        return []
        
    logger.info("Step 3/5: Verifying factual claims against reliable sources", num_claims=len(claims))
    
    verification_results = []
    
    async with AsyncCodewordsClient() as client:
        for claim in claims[:5]:  # Limit to 5 claims to avoid timeout
            # Search for claim verification on reliable sources
            search_queries = [
                f'{claim} site:reuters.com OR site:bbc.com OR site:ap.org',
                f'{claim} site:snopes.com OR site:politifact.com OR site:factcheck.org',
                f'"{claim}" fact check verification'
            ]
            
            supporting_sources = []
            disputing_sources = []
            
            for query in search_queries:
                try:
                    response = await client.run(
                        service_id="searchapi",
                        path="/search",
                        inputs={
                            "engine": "google",
                            "query": query,
                            "num_results": 3
                        }
                    )
                    response.raise_for_status()
                    search_results = response.json()
                    
                    for result in search_results.get("organic_results", []):
                        source_url = result.get("link", "")
                        if any(domain in source_url for domain in ["reuters.com", "bbc.com", "ap.org", "snopes.com", "politifact.com", "factcheck.org"]):
                            supporting_sources.append(source_url)
                            
                except Exception as e:
                    logger.warning("Search query failed", query=query, error=str(e))
                    continue
            
            # Determine verification status based on sources found
            if supporting_sources:
                status = "supported" if len(supporting_sources) >= 2 else "unclear"
            else:
                status = "no_evidence"
                
            verification_results.append(VerificationResult(
                claim=claim,
                verification_status=status,
                supporting_sources=supporting_sources[:3],  # Limit to top 3
                disputing_sources=disputing_sources[:3],
                evidence_summary=f"Found {len(supporting_sources)} sources discussing this claim"
            ))
    
    logger.debug("Completed claim verification", num_verified=len(verification_results))
    return verification_results

async def _generate_credibility_report(content: dict, analysis: ArticleAnalysis, verification_results: list[VerificationResult]) -> str:
    """Generate comprehensive credibility assessment report."""
    logger.info("Step 4/5: Generating detailed credibility report", title=content["title"])

    # Handle opinion articles differently
    if analysis.article_type in ["opinion", "editorial"]:
        return f"""# Article Analysis Report

## Article Classification
**Type:** {analysis.article_type.title()} Piece
**Source:** {content['source_domain']}
**Title:** {content['title']}

## Assessment Result
🟡 **This is an {analysis.article_type} piece** - Opinion content is not subject to fact-checking as it represents the author's viewpoint rather than factual claims.

## Analysis Summary
- **Content Type:** {analysis.article_type.title()}
- **Bias Indicators Found:** {len(analysis.bias_indicators) if analysis.bias_indicators else 0}
- **Writing Quality Score:** {analysis.credibility_score}/10

## Recommendation
Treat this as editorial content expressing opinions and perspectives rather than factual news reporting. Consider the author's expertise and potential biases when evaluating the arguments presented.
"""

    # Build verification summary
    verification_text = ""
    if verification_results:
        verification_text = "\n=== FACT VERIFICATION RESULTS ===\n"
        for i, result in enumerate(verification_results, 1):
            verification_text += f"\nClaim {i}: {result.claim}\n"
            verification_text += f"Status: {result.verification_status}\n"
            verification_text += f"Supporting Sources: {len(result.supporting_sources)}\n"
            verification_text += f"Evidence: {result.evidence_summary}\n"
            if result.supporting_sources:
                verification_text += f"Sources: {', '.join(result.supporting_sources[:2])}\n"
    
    bias_text = "\n=== BIAS ANALYSIS ===\n"
    if analysis.bias_indicators:
        bias_text += "\n".join(f"- {indicator}" for indicator in analysis.bias_indicators)
    else:
        bias_text += "No significant bias indicators detected."

    prompt = dedent("""\
        Generate a comprehensive fake news detection report based on the following analysis:
        
        <article_info>
        Title: {title}
        Source: {source_domain}
        Article Type: {article_type}
        Initial Credibility Score: {credibility_score}/10
        </article_info>

        {verification_text}
        
        {bias_text}

        Provide a detailed report in markdown format with the following structure:

        # Fake News Detection Report

        ## Overall Assessment
        [Provide clear verdict: Reliable/Questionable/Likely False with confidence level]

        ## Article Classification
        [Explain the type of content and its journalistic standards]

        ## Fact Verification Summary
        [Summarize which claims were verified, supported, or disputed]

        ## Source Credibility Analysis
        [Assess the credibility of the publication and domain]

        ## Bias and Manipulation Indicators
        [Detail any concerning patterns, emotional manipulation, or bias detected]

        ## Evidence-Based Assessment
        [Explain reasoning behind the overall verdict with specific evidence]

        ## Verification Sources
        [List the reliable sources used for fact-checking with links]

        ## Recommendation
        [Clear advice on how users should treat this information]
        
        Focus on providing clear, evidence-based assessment. Be specific about what was verified and what remains unclear.\
        """).format(
        title=content["title"],
        source_domain=content["source_domain"],
        article_type=analysis.article_type,
        credibility_score=analysis.credibility_score,
        verification_text=verification_text,
        bias_text=bias_text
    )

    client = AsyncOpenAI(base_url=urljoin(os.environ["CODEWORDS_RUNTIME_URI"], "run/gemini/v1"))
    response = await client.chat.completions.create(
        model="gemini-2.5-pro",
        messages=[
            {"role": "system", "content": "You are an expert fact-checker and misinformation analyst who provides clear, evidence-based assessments of news content credibility."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    logger.debug("Generated credibility report", title=content["title"], report_length=len(response.choices[0].message.content))
    return response.choices[0].message.content.strip()

async def detect_fake_news(url: str) -> str:
    """Main function to analyze article credibility and detect potential fake news."""
    logger.info("Starting fake news detection analysis", url=url)
    
    # Step 1: Extract article content
    content = await _extract_article_content(url)
    
    # Step 2: Analyze content and classify article type
    analysis = await _analyze_article_content(content)
    
    # Step 3: Verify factual claims (skip for opinion pieces)
    if analysis.article_type in ["opinion", "editorial"]:
        verification_results = []
        logger.info("Skipping fact verification for opinion piece", article_type=analysis.article_type)
    else:
        verification_results = await _verify_factual_claims(analysis.factual_claims, content["title"])
    
    # Step 4: Generate comprehensive credibility report
    report = await _generate_credibility_report(content, analysis, verification_results)
    
    logger.info("Completed fake news detection analysis", url=url)
    return report

# -------------------------
# FastAPI Application
# -------------------------
app = FastAPI(
    title="Fake News Detection Tool",
    description="Analyze news articles for credibility and detect potential misinformation using AI-powered fact-checking",
    version="1.0.0",
)

class FakeNewsDetectionRequest(BaseModel):
    url: str = Field(
        ..., 
        description="URL of the news article to analyze", 
        example="https://www.reuters.com/technology/artificial-intelligence/",
        min_length=10
    )

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        url = v.strip()
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return url

class FakeNewsDetectionResponse(BaseModel):
    url: str = Field(..., description="The analyzed article URL")
    credibility_report: str = Field(
        ..., 
        description="Comprehensive credibility analysis report in markdown format", 
        json_schema_extra={"contentMediaType": "text/markdown"}
    )

@app.post("/", response_model=FakeNewsDetectionResponse)
async def detect_fake_news_endpoint(request: FakeNewsDetectionRequest):
    """
    Analyze a news article for credibility and detect potential fake news.

    - **url**: Complete URL of the news article to analyze

    Returns comprehensive credibility analysis including:
    - Overall reliability assessment (Reliable/Questionable/Likely False)
    - Article type classification (news/opinion/editorial)
    - Fact verification results from authoritative sources
    - Bias and manipulation indicators
    - Source credibility analysis
    - Evidence-based reasoning for the assessment
    - Actionable recommendations
    """
    logger.info("Processing fake news detection request", url=request.url)

    return FakeNewsDetectionResponse(
        url=request.url,
        credibility_report=await detect_fake_news(url=request.url),
    )

if __name__ == "__main__":
    run_service(app)
