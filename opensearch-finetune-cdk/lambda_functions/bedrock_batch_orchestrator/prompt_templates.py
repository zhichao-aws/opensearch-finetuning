"""
Prompt templates for query generation with Bedrock
"""

QUERY_GENERATION_PROMPT = """You are an expert at generating search queries.

Given the following document, generate 5 diverse search queries that a user might use to find this document.

Document:
{document_text}

Generate exactly 5 queries in the following JSON format:
{{"queries": ["query1", "query2", "query3", "query4", "query5"]}}

Only return the JSON, no additional text."""


def format_prompt_for_document(document_text: str) -> str:
    """Format the query generation prompt for a document."""
    # Truncate document if too long (to fit in context window)
    max_length = 2000
    if len(document_text) > max_length:
        document_text = document_text[:max_length] + "..."

    return QUERY_GENERATION_PROMPT.format(document_text=document_text)
