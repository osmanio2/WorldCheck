# 🕵️ Fake News Detection Tool
An AI-powered fake news detection service that analyzes articles for credibility using advanced content analysis, multi-source fact-checking, and comprehensive bias detection.

🎯 Features
- Smart Content Classification - Automatically distinguishes between news articles, opinion pieces, and editorials
- Multi-Source Fact Verification - Cross-references claims against authoritative sources (Reuters, BBC, AP News, Snopes, PolitiFact, FactCheck.org)
- AI-Powered Analysis - Uses Google's Gemini 2.5 Pro for sophisticated content analysis and claim extraction
- Bias Detection - Identifies manipulation indicators, emotional language, and potential misinformation techniques
- Evidence-Based Assessments - Provides detailed reasoning with supporting evidence and source links
- Opinion-Aware Processing - Appropriately handles editorial content without false fact-checking
🚀 Quick Start
- CodeWords Workflow: https://codewords.agemo.ai/run/fake_news_detector_26fdfe66
- Full Webapp: https://v0-fake-news-detector-three.vercel.app/

📋 How It Works
- Content Extraction - Uses Firecrawl to extract clean article content from any URL
- AI Classification - Gemini 2.5 Pro analyzes content type and extracts verifiable factual claims
- Smart Routing - Opinion pieces are flagged appropriately; news articles proceed to fact-checking
- Multi-Source Verification - Claims are cross-referenced against trusted fact-checking sources
- Comprehensive Report - Generates detailed credibility assessment with evidence and reasoning
