Company Search Engine

A high‑performance company search app built with Streamlit. It combines BM25 keyword search, compressed semantic embeddings (FAISS + PCA), structured filters, and cross‑encoder reranking to deliver precise results from natural language queries.
Features

    🔍 Natural Language Parsing – Extracts filters like country, employee range, revenue, industry, etc. from plain text.

    ⚡ Hybrid Retrieval – BM25 + semantic search for broad coverage.

    🎯 Structured Filtering – Filter by country, public/private, revenue, employees, year, NAICS, business models, etc.

    🔁 Result Fusion – Reciprocal Rank Fusion (RRF) combines BM25 and semantic rankings.

    🚀 Reranking – Cross‑encoder re‑ranks top candidates for high precision.

    📊 Interactive UI – Clean interface with filter badges and responsive data table.

    🔧 Filter Relaxation – Automatically drops non‑critical filters if no results are found.

How It Works (Brief)

    Data Prep – Reads companies.txt (JSON lines), parses nested fields, deduplicates, and builds a combined text field.

    Indexing

        BM25 index on combined text.

        Semantic index: all-MiniLM-L6-v2 embeddings → optional PCA compression → FAISS IVF/HNSW index.

        Structured index: inverted indexes (sets/Roaring bitmaps) + segment trees for numeric ranges.

    Query – User types a query; parser extracts filters (regex + keyword maps).

    Retrieval – BM25 and semantic search each return top 200 candidates.

    Filtering – Structured filters applied; if no results, relax filters stepwise.

    Fusion – RRF merges filtered results into top 100.

    Reranking – Cross‑encoder scores candidates, returns final top 10.

    Display – Shows filters as badges, results table, and search time.

Installation
bash

pip install streamlit pandas numpy torch nltk rank-bm25 sentence-transformers faiss-cpu scikit-learn tqdm
# Optional: pip install pyroaring  # faster set operations

Place your company data in companies.txt (one JSON object per line).
Run: streamlit run app.py
Usage Example

Query: “logistics companies in Romania with more than 100 employees”
The app extracts: country: ro, employee_min: 100, business_models: logistics.
Results appear in a table with rank, name, country, and description snippet.
Key Components
Component	Role
BM25Search	Keyword retrieval
CompressedSemanticSearch	Semantic retrieval (PCA + FAISS)
StructuredFilter	Fast filtering (inverted indexes + segment trees)
Fusion	RRF combination
CrossEncoderReranker	Final reranking
parse_query	NL filter extraction
Performance Notes

    PCA compression (384→128 dim) speeds up FAISS with minimal accuracy loss.

    FAISS IVF index for large datasets; falls back to HNSW for smaller ones.

    Roaring bitmaps (pyroaring) accelerate set operations if installed.

    All indices cached via @st.cache_resource – subsequent runs are fast.

Data Format (Required Fields)
Field	Type	Description
website	string	Used for ID and deduplication
operational_name	string	Company name
description	string	Main search text
address	object/string	Must contain country_code
revenue	number	For range filters
employee_count	number	
year_founded	number	
is_public	boolean	
business_model	array/string	
target_markets	array/string	
core_offerings	array/string	
primary_naics	object/string	Optional – used for NAICS filtering
secondary_naics	array/object	Optional
