import streamlit as st
import pandas as pd
import numpy as np
import torch
import nltk
import re
import json
import ast
import pickle
import os
import hashlib
import time
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from sklearn.decomposition import PCA

# Optional: Roaring bitmaps for ultra-fast set operations
try:
    import pyroaring as pr
    HAS_ROARING = True
except ImportError:
    HAS_ROARING = False
    # We'll just use sets silently

# Ensure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ----------------------------------------------------------------------
# 1. Helper functions for data cleaning (from your notebook)
# ----------------------------------------------------------------------
def parse_address(address_field):
    if pd.isna(address_field):
        return None
    if isinstance(address_field, dict):
        addr_dict = address_field
    elif isinstance(address_field, str):
        try:
            addr_dict = json.loads(address_field.replace("'", '"'))
        except:
            try:
                addr_dict = ast.literal_eval(address_field)
            except:
                return None
    else:
        return None
    return addr_dict.get('country_code') or addr_dict.get('country_name')

def parse_naics(naics_field):
    if pd.isna(naics_field):
        return None
    if isinstance(naics_field, dict):
        return naics_field
    if isinstance(naics_field, str):
        try:
            return json.loads(naics_field.replace("'", '"'))
        except:
            try:
                return ast.literal_eval(naics_field)
            except:
                return None
    if isinstance(naics_field, list):
        return [parse_naics(item) for item in naics_field if item is not None]
    return None

def deduplicate_companies(df):
    df['_dup_key'] = df['website'].fillna('') + '_' + df['country'].fillna('') + '_' + df['operational_name'].fillna('')
    def choose_best(group):
        non_null_count = group.notna().sum(axis=1)
        return group.loc[non_null_count.idxmax()]
    df_dedup = df.groupby('_dup_key').apply(choose_best, include_groups=False).reset_index(drop=True)
    return df_dedup

def combine_text(row):
    parts = []
    
    def is_present(val):
        if val is None:
            return False
        if isinstance(val, float) and pd.isna(val):
            return False
        return True

    if is_present(row.get('description')):
        parts.append(str(row['description']))
    
    if is_present(row.get('core_offerings')):
        if isinstance(row['core_offerings'], list):
            parts.append(' '.join(row['core_offerings']))
        else:
            parts.append(str(row['core_offerings']))
    
    if is_present(row.get('target_markets')):
        if isinstance(row['target_markets'], list):
            parts.append(' '.join(row['target_markets']))
        else:
            parts.append(str(row['target_markets']))
    
    if is_present(row.get('operational_name')):
        parts.append(str(row['operational_name']))
    
    return ' '.join(parts)

# ----------------------------------------------------------------------
# 2. Segment Tree (for numeric range queries)
# ----------------------------------------------------------------------
class SegmentTree:
    def __init__(self, pairs: List[Tuple[float, Union[int, str]]], use_bitmap: bool = False):
        if not pairs:
            self.n = 0
            self.tree = []
            return
        pairs.sort(key=lambda x: x[0])
        self.values = [v for v, _ in pairs]
        self.ids = [id for _, id in pairs]
        self.n = len(pairs)
        self.use_bitmap = use_bitmap and HAS_ROARING
        self.tree = [None] * (4 * self.n)
        self._build(1, 0, self.n - 1)

    def _build(self, node, left, right):
        if left == right:
            if self.use_bitmap:
                self.tree[node] = pr.BitMap([self.ids[left]])
            else:
                self.tree[node] = {self.ids[left]}
        else:
            mid = (left + right) // 2
            self._build(node * 2, left, mid)
            self._build(node * 2 + 1, mid + 1, right)
            if self.use_bitmap:
                self.tree[node] = self.tree[node * 2] | self.tree[node * 2 + 1]
            else:
                self.tree[node] = self.tree[node * 2].union(self.tree[node * 2 + 1])

    def query_range(self, min_val: Optional[float], max_val: Optional[float]) -> Set[str]:
        if self.n == 0:
            return set()
        if min_val is None and max_val is None:
            return set(self.ids) if not self.use_bitmap else set(self.tree[1])
        lo = 0
        hi = self.n - 1
        if min_val is not None:
            lo = bisect.bisect_left(self.values, min_val)
        if max_val is not None:
            hi = bisect.bisect_right(self.values, max_val) - 1
        if lo > hi:
            return set()
        result = self._query(1, 0, self.n - 1, lo, hi)
        if self.use_bitmap:
            return set(result)
        return result

    def _query(self, node, left, right, ql, qr):
        if ql <= left and right <= qr:
            return self.tree[node]
        mid = (left + right) // 2
        if qr <= mid:
            return self._query(node * 2, left, mid, ql, qr)
        elif ql > mid:
            return self._query(node * 2 + 1, mid + 1, right, ql, qr)
        else:
            left_res = self._query(node * 2, left, mid, ql, qr)
            right_res = self._query(node * 2 + 1, mid + 1, right, ql, qr)
            if self.use_bitmap:
                return left_res | right_res
            else:
                return left_res.union(right_res)

# ----------------------------------------------------------------------
# 3. BM25 Search
# ----------------------------------------------------------------------
class BM25Search:
    def __init__(self, companies: List[Dict[str, Any]]):
        self.companies = companies
        self.ids = [c.get('id', str(i)) for i, c in enumerate(companies)]
        self.texts = [c.get('full_text_for_embedding', '') for c in companies]
        self.tokenized_corpus = [self._tokenize(doc) for doc in tqdm(self.texts, desc="BM25 tokenization", disable=True)]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text.lower())
        except:
            return text.lower().split()

    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.ids[idx], float(scores[idx])))
        return results

# ----------------------------------------------------------------------
# 4. Compressed Semantic Search (PCA + FAISS)
# ----------------------------------------------------------------------
class CompressedSemanticSearch:
    def __init__(
        self,
        companies: List[Dict[str, Any]],
        model_name: str = "all-MiniLM-L6-v2",
        compress_dim: Optional[int] = 128,
        use_ivf: bool = True,
        nlist: int = 1000,
        nprobe: int = 10,
        index_path: Optional[str] = "compressed.index",
        metadata_path: Optional[str] = "compressed_meta.pkl",
        use_gpu: bool = True,
        batch_size: int = 64,
    ):
        self.companies = companies
        self.model_name = model_name
        self.compress_dim = compress_dim
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.nprobe = nprobe
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.batch_size = batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.orig_dim = self.model.get_sentence_embedding_dimension()

        self.pca = None
        if compress_dim:
            pass  # just a placeholder

        if index_path and os.path.exists(index_path) and metadata_path and os.path.exists(metadata_path):
            self._load_index()
        else:
            self._build_index()

    def _build_index(self):
        texts = [c.get("full_text_for_embedding", "") for c in self.companies]
        ids = [c.get("id", str(i)) for i, c in enumerate(self.companies)]
        embeddings = self._encode_batch(texts)
        embeddings = np.ascontiguousarray(embeddings)

        if self.compress_dim:
            self.pca = PCA(n_components=self.compress_dim, whiten=False)
            embeddings = self.pca.fit_transform(embeddings).astype(np.float32)
            embeddings = np.ascontiguousarray(embeddings)

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        if self.use_ivf and len(embeddings) > 10000:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.nprobe = self.nprobe
        else:
            index = faiss.IndexHNSWFlat(dim, 16)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64

        index.add(embeddings)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self.index = index
        self.ids = ids
        self.dim = dim
        self._save_index()

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            with torch.no_grad():
                emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings).astype(np.float32)

    def _save_index(self):
        index_to_save = self.index
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(index_to_save, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "ids": self.ids,
                "dim": self.dim,
                "pca": self.pca,
                "orig_dim": self.orig_dim,
                "compress_dim": self.compress_dim,
            }, f)

    def _load_index(self):
        index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            meta = pickle.load(f)
        self.ids = meta["ids"]
        self.dim = meta["dim"]
        self.pca = meta["pca"]
        self.orig_dim = meta["orig_dim"]
        self.compress_dim = meta["compress_dim"]
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self.index = index

    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        q_emb = self.model.encode(query, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
        if self.pca:
            q_emb = self.pca.transform(q_emb)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.ids[idx], float(score)))
        return results

# ----------------------------------------------------------------------
# 5. Cross-Encoder Reranker
# ----------------------------------------------------------------------
class CrossEncoderReranker:
    def __init__(self, companies: List[Dict[str, Any]], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32):
        self.companies = companies
        self.company_dict = {c.get('id', str(i)): c for i, c in enumerate(companies)}
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, candidates: List[Tuple[str, float]], top_n: int = 10) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        pairs = []
        valid_ids = []
        for doc_id, _ in candidates:
            company = self.company_dict.get(doc_id)
            if company is None:
                continue
            text = company.get('full_text_for_embedding', '') or company.get('description', '')
            pairs.append([query, text])
            valid_ids.append(doc_id)
        if not pairs:
            return []
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        results = [(valid_ids[i], float(scores[i])) for i in range(len(valid_ids))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

# ----------------------------------------------------------------------
# 6. Structured Filter (with Roaring Bitmaps)
# ----------------------------------------------------------------------
class StructuredFilter:
    def __init__(self, companies: List[Dict[str, Any]], use_bitmap: bool = True):
        self.companies = companies
        self.use_bitmap = use_bitmap and HAS_ROARING
        self._rebuild_indexes()

    def _rebuild_indexes(self):
        if self.use_bitmap:
            self.country_index = defaultdict(pr.BitMap)
            self.public_ids = pr.BitMap()
            self.private_ids = pr.BitMap()
            self.business_model_index = defaultdict(pr.BitMap)
            self.target_market_index = defaultdict(pr.BitMap)
            self.core_offering_index = defaultdict(pr.BitMap)
        else:
            self.country_index = defaultdict(set)
            self.public_ids = set()
            self.private_ids = set()
            self.business_model_index = defaultdict(set)
            self.target_market_index = defaultdict(set)
            self.core_offering_index = defaultdict(set)

        revenue_pairs = []
        employee_pairs = []
        year_pairs = []

        self.id_to_company = {}
        self.id_to_index = {}
        self.id_to_int = {}
        self.int_to_id = {}

        for idx, comp in enumerate(self.companies):
            cid = comp['id']
            self.id_to_company[cid] = comp
            self.id_to_index[cid] = idx
            int_id = idx
            self.id_to_int[cid] = int_id
            self.int_to_id[int_id] = cid

            if country := comp.get('country'):
                if self.use_bitmap:
                    self.country_index[country.lower()].add(int_id)
                else:
                    self.country_index[country.lower()].add(cid)

            is_pub = comp.get('is_public')
            if is_pub is True:
                if self.use_bitmap:
                    self.public_ids.add(int_id)
                else:
                    self.public_ids.add(cid)
            elif is_pub is False:
                if self.use_bitmap:
                    self.private_ids.add(int_id)
                else:
                    self.private_ids.add(cid)

            for bm in comp.get('business_model') or []:
                if self.use_bitmap:
                    self.business_model_index[bm.lower()].add(int_id)
                else:
                    self.business_model_index[bm.lower()].add(cid)

            for tm in comp.get('target_markets') or []:
                if self.use_bitmap:
                    self.target_market_index[tm.lower()].add(int_id)
                else:
                    self.target_market_index[tm.lower()].add(cid)

            for co in comp.get('core_offerings') or []:
                if self.use_bitmap:
                    self.core_offering_index[co.lower()].add(int_id)
                else:
                    self.core_offering_index[co.lower()].add(cid)

            if (rev := comp.get('revenue')) is not None:
                revenue_pairs.append((rev, int_id))
            if (emp := comp.get('employee_count')) is not None:
                employee_pairs.append((emp, int_id))
            if (yr := comp.get('year_founded')) is not None:
                year_pairs.append((yr, int_id))

        self.revenue_tree = SegmentTree(revenue_pairs, use_bitmap=self.use_bitmap)
        self.employee_tree = SegmentTree(employee_pairs, use_bitmap=self.use_bitmap)
        self.year_tree = SegmentTree(year_pairs, use_bitmap=self.use_bitmap)

        self.deleted_ids = set()

    def _range_ids(self, min_val=None, max_val=None, field='revenue') -> Set[str]:
        if field == 'revenue':
            int_ids = self.revenue_tree.query_range(min_val, max_val)
        elif field == 'employee':
            int_ids = self.employee_tree.query_range(min_val, max_val)
        elif field == 'year':
            int_ids = self.year_tree.query_range(min_val, max_val)
        else:
            raise ValueError(f"Unknown field: {field}")
        return {self.int_to_id[i] for i in int_ids if i in self.int_to_id}

    def _values_to_ids(self, index, values: Optional[Union[str, List[str]]]):
        if values is None:
            return None
        if isinstance(values, str):
            values = [values]
        norm_vals = [v.lower() for v in values]
        if self.use_bitmap:
            result = pr.BitMap()
            for v in norm_vals:
                result |= index.get(v, pr.BitMap())
            return {self.int_to_id[i] for i in result}
        else:
            result = set()
            for v in norm_vals:
                result.update(index.get(v, set()))
            return result

    def _naics_matches(self, comp: Dict, naics_codes: List[str]) -> bool:
        codes = []
        if comp.get('primary_naics') and isinstance(comp['primary_naics'], dict):
            codes.append(comp['primary_naics'].get('code', ''))
        for sec in comp.get('secondary_naics') or []:
            if isinstance(sec, dict):
                codes.append(sec.get('code', ''))
        codes = [c for c in codes if c]
        if not codes:
            return False
        for req in naics_codes:
            req_str = str(req)
            if any(c.startswith(req_str) for c in codes):
                return True
        return False

    def filter(
        self,
        *,
        country: Optional[Union[str, List[str]]] = None,
        revenue_min: Optional[float] = None,
        revenue_max: Optional[float] = None,
        employee_min: Optional[int] = None,
        employee_max: Optional[int] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        is_public: Optional[bool] = None,
        naics_codes: Optional[List[str]] = None,
        business_models: Optional[List[str]] = None,
        target_markets: Optional[List[str]] = None,
        core_offerings: Optional[List[str]] = None,
        skip_missing: bool = False,
        return_indices: bool = False
    ) -> Union[List[int], List[Dict]]:
        candidate_sets = []

        if country is not None:
            ids = self._values_to_ids(self.country_index, country)
            if ids is not None:
                candidate_sets.append(ids)
        if is_public is not None:
            if self.use_bitmap:
                int_set = self.public_ids if is_public else self.private_ids
                ids = {self.int_to_id[i] for i in int_set}
            else:
                ids = self.public_ids if is_public else self.private_ids
            candidate_sets.append(ids)
        if business_models is not None:
            ids = self._values_to_ids(self.business_model_index, business_models)
            if ids is not None:
                candidate_sets.append(ids)
        if target_markets is not None:
            ids = self._values_to_ids(self.target_market_index, target_markets)
            if ids is not None:
                candidate_sets.append(ids)
        if core_offerings is not None:
            ids = self._values_to_ids(self.core_offering_index, core_offerings)
            if ids is not None:
                candidate_sets.append(ids)

        rev_ids = self._range_ids(revenue_min, revenue_max, field='revenue')
        if rev_ids is not None:
            candidate_sets.append(rev_ids)
        emp_ids = self._range_ids(employee_min, employee_max, field='employee')
        if emp_ids is not None:
            candidate_sets.append(emp_ids)
        year_ids = self._range_ids(year_min, year_max, field='year')
        if year_ids is not None:
            candidate_sets.append(year_ids)

        candidate_sets = [s for s in candidate_sets if s is not None]
        if not candidate_sets:
            all_ids = set(self.id_to_company.keys()) - self.deleted_ids
            result_ids = all_ids
        else:
            candidate_sets.sort(key=len)
            result_ids = candidate_sets[0].copy()
            for s in candidate_sets[1:]:
                result_ids.intersection_update(s)
                if not result_ids:
                    break
            result_ids -= self.deleted_ids

        if naics_codes and result_ids:
            final_ids = set()
            for cid in result_ids:
                comp = self.id_to_company[cid]
                if self._naics_matches(comp, naics_codes):
                    final_ids.add(cid)
            result_ids = final_ids

        if return_indices:
            return [self.id_to_index[cid] for cid in result_ids if cid in self.id_to_index]
        else:
            return [self.id_to_company[cid] for cid in result_ids if cid in self.id_to_company]

# ----------------------------------------------------------------------
# 7. Fusion (RRF)
# ----------------------------------------------------------------------
class Fusion:
    @staticmethod
    def reciprocal_rank_fusion(rankings: List[List[Tuple[str, float]]], k: int = 60, top_n: int = 100) -> List[Tuple[str, float]]:
        rrf_scores = {}
        for rank_list in rankings:
            for rank, (doc_id, _) in enumerate(rank_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]

# ----------------------------------------------------------------------
# 8. Query Parser (with country/industry maps)
# ----------------------------------------------------------------------
COUNTRY_MAP = {
    'romania': 'ro',
    'switzerland': 'ch',
    'united states': 'us',
    'usa': 'us',
    'u.s.': 'us',
    'america': 'us',
    'germany': 'de',
    'france': 'fr',
    'uk': 'gb',
    'united kingdom': 'gb',
    'spain': 'es',
    'italy': 'it',
    'netherlands': 'nl',
    'sweden': 'se',
    'norway': 'no',
    'denmark': 'dk',
    'finland': 'fi',
    'iceland': 'is',
    'europe': None,
    'scandinavia': ['se', 'no', 'dk', 'fi', 'is'],
    'asia': None,
    'north america': None,
}

INDUSTRY_KEYWORDS = {
    'logistic': 'logistics',
    'logistics': 'logistics',
    'transportation': 'transportation',
    'shipping': 'shipping',
    'freight': 'freight',
    'software': 'software',
    'saas': 'saas',
    'cloud': 'cloud',
    'hr tech': 'hr',
    'human resources': 'hr',
    'pharmaceutical': 'pharmaceutical',
    'pharma': 'pharmaceutical',
    'biotech': 'biotechnology',
    'biotechnology': 'biotechnology',
    'construction': 'construction',
    'contractor': 'construction',
    'engineering': 'engineering',
    'civil': 'construction',
    'fintech': 'fintech',
    'banking': 'banking',
    'payments': 'payments',
    'clean energy': 'clean energy',
    'renewable energy': 'renewable energy',
    'solar': 'solar',
    'wind': 'wind',
    'battery': 'battery',
    'electric vehicle': 'electric vehicle',
    'ev': 'electric vehicle',
    'food': 'food and beverage',
    'beverage': 'food and beverage',
    'packaging': 'packaging',
    'cosmetics': 'cosmetics',
    'beauty': 'cosmetics',
    'e-commerce': 'e-commerce',
    'ecommerce': 'e-commerce',
    'retail': 'retail',
    'shopify': 'e-commerce',
    'manufacturing': 'manufacturing',
    'manufacturer': 'manufacturing',
    'equipment': 'equipment',
    'machinery': 'equipment',
}

def parse_query(query: str) -> Dict[str, Any]:
    filters = {}
    query_lower = query.lower()

    for name, code in COUNTRY_MAP.items():
        if name in query_lower:
            if code is None:
                continue
            filters['country'] = code
            break

    if 'public' in query_lower:
        filters['is_public'] = True
    elif 'private' in query_lower:
        filters['is_public'] = False

    emp_match = re.search(r'(more than|over|>|≥|at least)\s*([\d,]+)\s*employees?', query_lower)
    if emp_match:
        filters['employee_min'] = int(emp_match.group(2).replace(',', ''))
    emp_match = re.search(r'(less than|under|<|≤|fewer than)\s*([\d,]+)\s*employees?', query_lower)
    if emp_match:
        filters['employee_max'] = int(emp_match.group(2).replace(',', ''))
    emp_match = re.search(r'([\d,]+)\s*employees?', query_lower)
    if emp_match and 'employee_min' not in filters and 'employee_max' not in filters:
        filters['employee_min'] = int(emp_match.group(1).replace(',', ''))

    rev_match = re.search(r'revenue\s*(over|>|≥|at least)?\s*\$?([\d,]+)\s*(million|billion|m|b)?', query_lower)
    if rev_match:
        val = float(rev_match.group(2).replace(',', ''))
        unit = rev_match.group(3)
        if unit in ('billion', 'b'):
            val *= 1_000_000_000
        elif unit in ('million', 'm'):
            val *= 1_000_000
        filters['revenue_min'] = val

    year_match = re.search(r'founded\s+(after|since|>|≥)\s*(\d{4})', query_lower)
    if year_match:
        filters['year_min'] = int(year_match.group(2))
    year_match = re.search(r'founded\s+(before|prior to|<|≤)\s*(\d{4})', query_lower)
    if year_match:
        filters['year_max'] = int(year_match.group(2))
    year_match = re.search(r'startups?\s+founded\s+after\s+(\d{4})', query_lower)
    if year_match:
        filters['year_min'] = int(year_match.group(1))

    tokens = set()
    for word, mapped in INDUSTRY_KEYWORDS.items():
        if word in query_lower:
            tokens.add(mapped)
    if tokens:
        filters['business_models'] = list(tokens)

    return filters

def apply_filters_with_relaxation(
    structured_filter: StructuredFilter,
    filter_kwargs: Dict[str, Any],
    bm25_results: List[Tuple[str, float]],
    sem_results: List[Tuple[str, float]]
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    if not filter_kwargs:
        return bm25_results, sem_results

    filter_priority = ['business_models', 'target_markets', 'core_offerings', 'country', 'is_public']
    current_filters = filter_kwargs.copy()

    while True:
        filtered_companies = structured_filter.filter(**current_filters)
        filtered_ids = {c['id'] for c in filtered_companies}
        bm25_f = [(cid, s) for cid, s in bm25_results if cid in filtered_ids]
        sem_f = [(cid, s) for cid, s in sem_results if cid in filtered_ids]

        if bm25_f or sem_f:
            return bm25_f, sem_f

        removed = False
        for key in filter_priority:
            if key in current_filters:
                # silently remove filter
                del current_filters[key]
                removed = True
                break
        if not removed:
            break
    return [], []

# ----------------------------------------------------------------------
# 9. Load data and initialize components (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_data_and_models():
    # Silently load everything – no progress messages in UI
    df = pd.read_json("companies.txt", lines=True)
    df['country'] = df['address'].apply(parse_address)
    df['primary_naics'] = df['primary_naics'].apply(parse_naics)
    df['secondary_naics'] = df['secondary_naics'].apply(parse_naics)
    df = deduplicate_companies(df)
    df['full_text_for_embedding'] = df.apply(combine_text, axis=1)
    df['id'] = df['website'].fillna('').astype(str) + '_' + df['operational_name'].fillna('').astype(str)
    df['id'] = df.apply(lambda row: row['id'] if row['id'].strip() else str(row.name), axis=1)
    companies = df.to_dict(orient='records')

    bm25 = BM25Search(companies)
    semantic = CompressedSemanticSearch(companies, compress_dim=128, use_ivf=True, nlist=1000, nprobe=10)
    reranker = CrossEncoderReranker(companies)
    structured_filter = StructuredFilter(companies, use_bitmap=True)

    return companies, bm25, semantic, reranker, structured_filter

# ----------------------------------------------------------------------
# 10. Streamlit UI – polished and user-friendly
# ----------------------------------------------------------------------
st.set_page_config(page_title="Company Search Engine", layout="wide")

# Custom CSS for a professional yet colorful look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    .result-table {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-country {
        background-color: #3498db20;
        color: #2980b9;
        border: 1px solid #3498db;
    }
    .badge-industry {
        background-color: #2ecc7120;
        color: #27ae60;
        border: 1px solid #2ecc71;
    }
    .badge-public {
        background-color: #f1c40f20;
        color: #f39c12;
        border: 1px solid #f1c40f;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 500;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔍 Company Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hello! I am here to help you find the perfect companies. Just type your query below.</div>', unsafe_allow_html=True)

# Load everything once (cached)
with st.spinner("Loading data and models – this may take a few minutes the first time..."):
    companies, bm25, semantic, reranker, structured_filter = load_data_and_models()

# Search area
with st.container():
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("", placeholder="e.g., Logistic companies in Romania with more than 100 employees", label_visibility="collapsed")
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if search_button and query:
    with st.spinner("Searching..."):
        start_time = time.perf_counter()

        # Parse query
        filter_kwargs = parse_query(query)

        # Retrieve candidates
        bm25_results = bm25.search(query, top_k=200)
        sem_results = semantic.search(query, top_k=200)

        # Apply structured filter with relaxation
        bm25_f, sem_f = apply_filters_with_relaxation(structured_filter, filter_kwargs, bm25_results, sem_results)

        # Fuse
        fused = Fusion.reciprocal_rank_fusion([bm25_f, sem_f], k=60, top_n=100)

        # Rerank
        final = reranker.rerank(query, fused, top_n=10)

        elapsed = time.perf_counter() - start_time

        # Show timing and extracted filters (optional, but can be displayed as badges)
        if filter_kwargs:
            # Convert filter dict into badges
            badge_html = '<div style="margin-bottom: 1rem;">'
            for k, v in filter_kwargs.items():
                if k == 'country':
                    badge_html += f'<span class="badge badge-country">🌍 {k}: {v}</span>'
                elif k == 'is_public':
                    badge_html += f'<span class="badge badge-public">🏢 {k}: {"public" if v else "private"}</span>'
                elif k in ['business_models', 'target_markets', 'core_offerings']:
                    if isinstance(v, list):
                        for item in v:
                            badge_html += f'<span class="badge badge-industry">🏷️ {item}</span>'
                else:
                    badge_html += f'<span class="badge badge-industry">{k}: {v}</span>'
            badge_html += '</div>'
            st.markdown(badge_html, unsafe_allow_html=True)

        st.markdown(f"⏱️ Found results in **{elapsed:.2f} seconds**")

        # Display results
        if not final:
            st.warning("No results found. Try broadening your query.")
        else:
            results_data = []
            for rank, (cid, _) in enumerate(final, 1):
                company = reranker.company_dict.get(cid, {})
                name = company.get('operational_name', 'Unknown')
                country = company.get('country', 'N/A')
                desc = company.get('description', '')
                # Trim description for display
                desc_short = desc[:200] + "..." if len(desc) > 200 else desc
                results_data.append({
                    "Rank": rank,
                    "Company": name,
                    "Country": country,
                    "Description": desc_short
                })
            df_results = pd.DataFrame(results_data)
            # Use a styled table for better appearance
            st.markdown('<div class="result-table">', unsafe_allow_html=True)
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "Company": st.column_config.TextColumn(width="medium"),
                    "Country": st.column_config.TextColumn(width="small"),
                    "Description": st.column_config.TextColumn(width="large"),
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)
