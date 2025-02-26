REWRITE_CLARIFY_QUERY_PROMPT = {
    'system_prompt': """
Please rewrite the given query to make it more specific, clear, and focused. Ensure that the revised query addresses the topic directly and eliminates any ambiguity. If necessary, add details or context to make the query more precise, so it can be better understood and answered. Output only the revised query in JSON format, with the query as the value of the 'query' key.

**Example Output (in JSON format):**
```json
{
  "query": "Your rewritten query goes here."
}
```""",
    'user_prompt': 'Original query: {query}'
}


REWRITE_EXPAND_QUERY_PROMPT = {
    'system_prompt': """
Please rewrite the given query by expanding it with additional relevant questions or variations that address the same topic. These expanded queries should help to explore different aspects or approaches related to the original query. You can introduce new perspectives, specific sub-questions, or related queries. Ensure the expanded queries are still coherent and closely related to the original question. Output only a JSON array, where each element is an expanded query as a string.

**Example Output (in JSON format):**
```json
[
  "Expanded query 1 goes here.",
  "Expanded query 2 goes here.",
  "Expanded query 3 goes here."
]
```""",
    'user_prompt': 'Original query: {query}'
}

REWRITE_CUSTOM_QUERY_PROMPT = {
    'system_prompt': """
Please rewrite the given query based on the following instruction: {instruction}. The rewritten query should align with the given directive and ensure clarity, specificity, and relevance to the original topic. Make sure the revised query is well-structured and can be easily understood, while following the instruction provided.

**Example Output (in JSON format):**
```json
{
  "query": "Your rewritten query goes here."
}
```""",
    'user_prompt': 'Original query: {query}'
}


DECOMPOSE_QUERY_PROMPT = {
    'system_prompt': """
Please split the given query into multiple smaller, more specific subqueries. Each subquery should focus on a different aspect of the original query and be concise and clear. The subqueries should be independent, but together they should still address the overall topic of the original query. Ensure that each subquery is well-formed, focused, and easy to understand. Output only a JSON array, where each element is a subquery as a string.

**Example Output (in JSON format):**
```json
[
  "Subquery 1 goes here.",
  "Subquery 2 goes here.",
  "Subquery 3 goes here."
]
```""",
    'user_prompt': 'Original query: {query}'
}

REFINE_DOC_SUMMARIZE_PROMPT = {
    'system_prompt': """
Please refine the given document to retain only the information helpful for answering the provided question. Remove any details or content that is irrelevant to the query. Ensure the refined document focuses on the key points that answer the query and is concise, clear, and easy to understand. The refined content should reflect the most relevant information while discarding any unnecessary or unrelated sections. Output only the refined content in JSON format, with the refined content as the value of the 'refined_content' key.

**Example Output (in JSON format):**
```json
{
  "refined_document": "Your refined content related to the question goes here."
}
```""",
    'user_prompt': 'Document: {document}\nQuestion: {question}'
}

REFINE_DOC_EXPLAIN_PROMPT =  {
    'system_prompt': """
Please read the given document carefully and think critically about it in relation to the provided question. Your task is to provide a detailed explanation that answers the question, using the information from the document. Your explanation should not only focus on directly answering the question but should also provide reasoning, analysis, and any insights that emerge from the document. Make sure your response is logical, clear, and thoroughly addresses the question based on the content of the document.

Output only the explanation in JSON format, with the explanation as the value of the 'explanation' key.

**Example Output (in JSON format):**
```json
{
  "explanation": "Your detailed explanation and reasoning based on the document goes here."
}
```""",
    'user_prompt': 'Original document: {document}\nQuestion: {question}'
}

