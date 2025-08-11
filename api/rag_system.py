# Domain-Agnostic RAG System - Fixed with Qdrant Authentication
# Works for ANY document type: medical, legal, technical, financial, etc.
# NOTE: pip install -U pymupdf4llm mammoth html2text qdrant-client fastapi uvicorn python-dotenv openai rank-bm25 cohere
from fastapi.middleware.cors import CORSMiddleware
import os, io, re, html, json, hashlib, uuid, requests, unicodedata, time, pickle, atexit
from typing import List, Union, Dict, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

import fitz  # PyMuPDF
import pymupdf4llm  # PDF -> Markdown (preserves headings/tables)
import mammoth       # DOCX -> Markdown
import html2text     # HTML -> Markdown

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------- BM25 for Hybrid Search ----------
from rank_bm25 import BM25Okapi

# ---------- Optional Cohere Reranker ----------
try:
    import cohere  # pip install cohere
except Exception:
    cohere = None

# ---------- Qdrant ----------
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny

# ---------- Configuration ----------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")  # 3072 dims
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4.1") 

# Reranker (optional)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
USE_RERANK = bool(COHERE_API_KEY)

# Retrieval sizes
CANDIDATES_PER_QUERY = int(os.getenv("CANDIDATES_PER_QUERY", "40"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "12"))
MAX_CTX_CHUNKS = int(os.getenv("MAX_CTX_CHUNKS", "8"))

# Rate limit prevention
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
EMBEDDING_BATCH_SIZE = 10
API_DELAY_MS = 200

# Other settings
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.12"))
RETURN_FORMAT = os.getenv("RETURN_FORMAT", "simple")
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"

# Qdrant settings - FIXED WITH API KEY
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # Get API key from environment
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "universal_rag_clean")

def infer_embedding_dim(model: str) -> int:
    m = model.lower()
    if "text-embedding-3-large" in m:
        return 3072
    if "text-embedding-3-small" in m:
        return 1536
    return int(os.getenv("EMBED_DIM", "3072"))

EMBED_DIM = infer_embedding_dim(EMBED_MODEL)
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY")

# Clients with timeout
client = OpenAI(timeout=30)
co = cohere.Client(COHERE_API_KEY) if (cohere and COHERE_API_KEY) else None

# ---------- Chunking Configuration ----------
md_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
    strip_headers=False,
)

md_chunker = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=400
)

app = FastAPI(title="Universal RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize Qdrant WITH AUTHENTICATION
print(f"Connecting to Qdrant at: {QDRANT_URL}")
if QDRANT_API_KEY:
    print("Using Qdrant API key authentication")
    qd = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=30
    )
else:
    print("No Qdrant API key provided, using without authentication")
    qd = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=30)

# ---------- Embedding Cache Management ----------
embedding_cache: Dict[str, np.ndarray] = {}

def load_embedding_cache():
    global embedding_cache
    try:
        with open('embedding_cache.pkl', 'rb') as f:
            embedding_cache = pickle.load(f)
            print(f"✓ Loaded {len(embedding_cache)} cached embeddings")
    except:
        embedding_cache = {}

def save_embedding_cache():
    try:
        with open('embedding_cache.pkl', 'wb') as f:
            pickle.dump(embedding_cache, f)
            print(f"✓ Saved {len(embedding_cache)} embeddings to cache")
    except Exception as e:
        print(f"Could not save embedding cache: {e}")

load_embedding_cache()
atexit.register(save_embedding_cache)

# ---------- Query Enhancement (Simple, Domain-Agnostic) ----------
class QueryEnhancer:
    """Simple query enhancement that works for any domain"""
    
    def generate_query_variants(self, query: str) -> List[Tuple[str, float]]:
        """Generate query variants without domain assumptions"""
        variants = [(query, 1.0)]  # Original with full weight
        
        # For compound questions, split them
        if " and " in query.lower() or ", " in query.lower():
            # Split on 'and' or comma to handle compound questions
            parts = re.split(r'\s+and\s+|,\s*', query, flags=re.IGNORECASE)
            if len(parts) > 1 and len(parts) <= 3:
                for part in parts:
                    part = part.strip()
                    if len(part) > 10:  # Only add meaningful parts
                        variants.append((part, 0.7))
        
        # For very long queries, create a simplified version
        if len(query.split()) > 20:
            # Extract likely key terms (capitalized words, quoted phrases, numbers)
            key_terms = []
            
            # Find quoted phrases
            quoted = re.findall(r'"([^"]+)"', query)
            key_terms.extend(quoted)
            
            # Find capitalized words (likely proper nouns/important terms)
            words = query.split()
            caps = [w for w in words if w[0].isupper() and len(w) > 2]
            key_terms.extend(caps[:5])  # Limit to avoid noise
            
            # Find numbers/amounts/percentages
            numbers = re.findall(r'\b\d+\.?\d*%?\b', query)
            key_terms.extend(numbers)
            
            if key_terms:
                simplified = " ".join(key_terms)
                if simplified != query and len(simplified) > 5:
                    variants.append((simplified, 0.5))
        
        return variants[:3]  # Maximum 3 variants

query_enhancer = QueryEnhancer()

# ---------- Helpers ----------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---------- Text Cleaning ----------
def clean_for_chunking(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\ufeff', '').replace('\x00', '')
    text = text.replace('\r\n', '\n')
    return text

def clean_for_embedding(text: str) -> str:
    if not text:
        return ""
    text = (
        text.replace("\ufeff", "")
            .replace("\u200b", "")
            .replace("\u200c", "")
            .replace("\u200d", "")
            .replace("\u00a0", " ")
            .replace("\xa0", " ")
            .replace("–", "-")
            .replace("—", "-")
    )
    # Smart quotes -> ASCII
    text = text.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch if ch.isprintable() or ch in "\n\t\r" else " " for ch in text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\t+", " ", text)
    return text.strip()

# ---------- Document Processing ----------
def fetch(url: str) -> bytes:
    try:
        print(f"Fetching: {url[:100]}...")
        # Simple direct request - no special headers
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        print(f"Downloaded {len(r.content)} bytes")
        return r.content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        raise

def detect_document_type(url_lower: str, data: bytes) -> str:
    url_clean = url_lower.split("?")[0]
    if ".pdf" in url_clean or data[:4] == b"%PDF":
        return "pdf"
    if ".docx" in url_clean or data[:4] == b"PK\x03\x04":
        return "docx"
    if any(x in url_clean for x in [".txt", ".html", ".htm"]):
        return "text"
    if data[:5].lower() == b"<html":
        return "text"
    return "text"

def md_to_chunks(md: str, origin: str):
    md = clean_for_chunking(md)
    header_docs = md_header_splitter.split_text(md)
    parts = []
    for d in header_docs:
        parts.extend(md_chunker.split_text(d.page_content))
    out, meta = [], []
    for i, piece in enumerate(parts):
        if len(piece.strip()) < 50:
            continue
        out.append(piece)
        meta.append({
            "source": origin,
            "page": None,
            "chunk_index": i,  # Simple index instead of classification
        })
    return out, meta

def chunks_from_pdf(data: bytes, origin: str):
    if data[:4] != b"%PDF":
        idx = data.find(b"%PDF")
        if 0 < idx < 1024:
            data = data[idx:]
        else:
            raise ValueError("Not a valid PDF file")
    doc = fitz.open(stream=data, filetype="pdf")
    md = pymupdf4llm.to_markdown(doc, write_images=False)
    doc.close()
    return md_to_chunks(md, origin)

def chunks_from_docx(data: bytes, origin: str):
    md = mammoth.convert_to_markdown(io.BytesIO(data)).value
    return md_to_chunks(md, origin)

def chunks_from_text(data: bytes, origin: str):
    try:
        txt = data.decode("utf-8", errors="ignore")
    except Exception:
        txt = data.decode("latin-1", errors="ignore")
    if "<html" in txt.lower():
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        md = h.handle(txt)
    else:
        md = txt
    return md_to_chunks(md, origin)

def extract_chunks(urls: List[str]):
    chunks, metas = [], []
    for url in urls:
        data = fetch(url)
        doc_type = detect_document_type(url.lower(), data)
        if doc_type == "pdf":
            c, m = chunks_from_pdf(data, url)
        elif doc_type == "docx":
            c, m = chunks_from_docx(data, url)
        else:
            c, m = chunks_from_text(data, url)
        if c:
            chunks.extend(c)
            metas.extend(m)
            print(f"✓ Extracted {len(c)} chunks from {doc_type}")
    print(f"Total chunks extracted: {len(chunks)}")
    return chunks, metas

# ---------- Embeddings with Cache + Retry ----------
def embed_texts_batch_with_retry(texts: List[str], use_clean: bool = True, max_retries: int = 3) -> np.ndarray:
    if not texts:
        return np.array([])
    if use_clean:
        texts = [clean_for_embedding(t) for t in texts]

    embeddings = [None] * len(texts)
    to_embed, idxs = [], []

    for i, t in enumerate(texts):
        key = hashlib.md5(t.encode()).hexdigest()
        if key in embedding_cache:
            embeddings[i] = embedding_cache[key]
        else:
            to_embed.append(t)
            idxs.append(i)

    if not to_embed:
        return np.array(embeddings, dtype="float32")

    for batch_start in range(0, len(to_embed), EMBEDDING_BATCH_SIZE):
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(to_embed))
        batch_texts = to_embed[batch_start:batch_end]
        batch_idxs = idxs[batch_start:batch_end]
        retry_count = 0
        while retry_count < max_retries:
            try:
                if batch_start > 0:
                    time.sleep(API_DELAY_MS / 1000.0)
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
                newE = np.array([d.embedding for d in resp.data], dtype="float32")
                for i, vec in zip(batch_idxs, newE):
                    key = hashlib.md5(texts[i].encode()).hexdigest()
                    embedding_cache[key] = vec
                    embeddings[i] = vec
                break
            except Exception as e:
                if "rate" in str(e).lower():
                    retry_count += 1
                    wait_time = (2 ** retry_count) + np.random.uniform(0, 1)
                    print(f"Rate limit hit, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise e
    return np.array(embeddings, dtype="float32")

embed_texts_batch = embed_texts_batch_with_retry

# ---------- Hybrid Search (BM25 + Vector with RRF) ----------
class HybridSearcher:
    def __init__(self):
        self.bm25_index = None
        self.bm25_corpus: List[str] = []
        self.bm25_id_map: Dict[int, str] = {}
        self.chunk_id_to_payload: Dict[str, dict] = {}

    def build_bm25_index(self, chunks: List[str], chunk_ids: List[str], payloads: List[dict]):
        print("Building BM25 index...")
        tokenized = []
        for chunk in chunks:
            # Simple tokenization - works for any language/domain
            # Keep words, numbers, and common symbols
            tokens = re.findall(r"[a-zA-Z']{2,}|\d+(?:\.\d+)?%?|[€$£¥₹]?\d+[kKmMbB]?", chunk.lower())
            tokenized.append(tokens)
        self.bm25_index = BM25Okapi(tokenized)
        self.bm25_corpus = chunks
        self.bm25_id_map = {i: chunk_ids[i] for i in range(len(chunks))}
        self.chunk_id_to_payload = {chunk_ids[i]: payloads[i] for i in range(len(chunks))}
        try:
            with open('bm25_index_universal.pkl', 'wb') as f:
                pickle.dump({
                    'index': self.bm25_index,
                    'corpus': self.bm25_corpus,
                    'id_map': self.bm25_id_map,
                    'payload_map': self.chunk_id_to_payload
                }, f)
            print("✓ BM25 index saved")
        except Exception as e:
            print(f"Could not save BM25 index: {e}")

    def load_index(self) -> bool:
        try:
            with open('bm25_index_universal.pkl', 'rb') as f:
                data = pickle.load(f)
                self.bm25_index = data['index']
                self.bm25_corpus = data['corpus']
                self.bm25_id_map = data['id_map']
                self.chunk_id_to_payload = data.get('payload_map', {})
                print("✓ BM25 index loaded from disk")
                return True
        except:
            return False

    def hybrid_search(self, query: str, vector_results: List[dict], doc_ids: List[str], limit: int = 40) -> List[dict]:
        if not self.bm25_index:
            print("BM25 index not available, using vector search only")
            return vector_results[:limit]

        # Same tokenization as indexing
        query_tokens = re.findall(r"[a-zA-Z']{2,}|\d+(?:\.\d+)?%?|[€$£¥₹]?\d+[kKmMbB]?", query.lower())
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        bm25_results = []
        for idx, score in enumerate(bm25_scores):
            if score > 0:
                chunk_id = self.bm25_id_map.get(idx)
                if chunk_id and chunk_id in self.chunk_id_to_payload:
                    payload = self.chunk_id_to_payload[chunk_id]
                    if payload.get('doc_id') in doc_ids:
                        bm25_results.append({
                            'id': chunk_id,
                            'bm25_score': float(score),
                            'metadata': payload
                        })

        bm25_results.sort(key=lambda x: x['bm25_score'], reverse=True)
        bm25_results = bm25_results[:limit]

        # Reciprocal Rank Fusion (RRF)
        k = 60
        rrf_scores: Dict[str, dict] = {}

        for rank, result in enumerate(vector_results):
            doc_id = result['id']
            rrf_scores[doc_id] = {
                'score': 1 / (k + rank + 1),
                'metadata': result['metadata'],
                'vector_score': float(result.get('score', 0))
            }

        for rank, result in enumerate(bm25_results):
            doc_id = result['id']
            if doc_id in rrf_scores:
                rrf_scores[doc_id]['score'] += 1 / (k + rank + 1)
                rrf_scores[doc_id]['bm25_score'] = float(result['bm25_score'])
            else:
                rrf_scores[doc_id] = {
                    'score': 1 / (k + rank + 1),
                    'metadata': result['metadata'],
                    'bm25_score': float(result['bm25_score'])
                }

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True)

        final_results = []
        for doc_id, data in sorted_results[:limit]:
            final_results.append({
                'id': doc_id,
                'score': float(data['score']),
                'metadata': data['metadata'],
                'hybrid_score': float(data['score']),
                'vector_score': float(data.get('vector_score', 0)),
                'bm25_score': float(data.get('bm25_score', 0))
            })
        return final_results

hybrid_searcher = HybridSearcher()
hybrid_searcher.load_index()

# ---------- Qdrant Collection Management ----------
def init_qdrant_collection():
    try:
        # Try to check existing collections
        existing = [c.name for c in qd.get_collections().collections]
        if QDRANT_COLLECTION in existing:
            print(f"Deleting existing collection: {QDRANT_COLLECTION}")
            qd.delete_collection(collection_name=QDRANT_COLLECTION)
    except Exception as e:
        print(f"Error checking collections (might be authentication issue): {e}")
        # Try to create anyway

    try:
        print(f"Creating fresh collection: {QDRANT_COLLECTION}")
        qd.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print("✓ Qdrant collection initialized")
    except Exception as e:
        print(f"Error creating collection: {e}")
        print("Will attempt to use existing collection if available")

# Only initialize Qdrant if not skipped
if QDRANT_URL.lower() != "skip":
    try:
        init_qdrant_collection()
    except Exception as e:
        print(f"Warning: Could not initialize Qdrant collection: {e}")
        print("App will continue but Qdrant operations may fail")

# ---------- Upsert ----------
def upsert_chunks_to_qdrant(chunks: List[str], metas: List[dict]):
    if not chunks:
        return
    
    if QDRANT_URL.lower() == "skip":
        print("Qdrant skipped, not upserting chunks")
        return
        
    print(f"Upserting {len(chunks)} chunks to Qdrant...")

    chunk_ids = []
    payloads = []

    batch_size = 25
    global_idx_counter = 0
    for i in range(0, len(chunks), batch_size):
        if i > 0:
            time.sleep(0.5)

        Bc = chunks[i: i + batch_size]
        Bm = metas[i: i + batch_size]
        E = embed_texts_batch(Bc, use_clean=True)
        points = []

        for j, (text, meta) in enumerate(zip(Bc, Bm)):
            doc_id = sha1(meta["source"])
            raw = f"{doc_id}|{global_idx_counter}"
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

            payload = {
                "doc_id": doc_id,
                "source": meta["source"],
                "page": meta.get("page"),
                "text": text,
                "chunk_index": meta.get("chunk_index", global_idx_counter),
            }

            chunk_ids.append(pid)
            payloads.append(payload)
            points.append(PointStruct(id=pid, vector=E[j].tolist(), payload=payload))
            global_idx_counter += 1

        try:
            qd.upsert(collection_name=QDRANT_COLLECTION, points=points)
        except Exception as e:
            print(f"Error upserting to Qdrant: {e}")
            if "authentication" in str(e).lower() or "forbidden" in str(e).lower():
                print("Authentication error - check your Qdrant API key")

    print("✓ Qdrant upsert complete")

    if USE_HYBRID_SEARCH:
        hybrid_searcher.build_bm25_index(chunks, chunk_ids, payloads)

# ---------- Retrieval ----------
def qdrant_search_multi(queries: List[Tuple[str, float]], doc_ids: List[str], limit: int) -> List[dict]:
    """Search with multiple weighted queries"""
    if QDRANT_URL.lower() == "skip":
        return []
        
    all_results = {}
    
    for query_text, weight in queries:
        qv = client.embeddings.create(model=EMBED_MODEL, input=[clean_for_embedding(query_text)]).data[0].embedding
        
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))])
        
        try:
            res = qd.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=qv,
                limit=limit,
                query_filter=flt,
                with_payload=True,
                score_threshold=SCORE_THRESHOLD,
            )
            
            # If no results with threshold, try with lower threshold
            if not res and weight == 1.0:  # Only for main query
                res = qd.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=qv,
                    limit=limit,
                    query_filter=flt,
                    with_payload=True,
                    score_threshold=max(0.1, SCORE_THRESHOLD - 0.05),
                )
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []
        
        # Aggregate weighted results
        for r in res:
            if r.id in all_results:
                all_results[r.id]['score'] += float(r.score) * weight
            else:
                all_results[r.id] = {
                    "id": r.id,
                    "score": float(r.score) * weight,
                    "metadata": r.payload
                }
    
    # Sort by aggregate score
    results = list(all_results.values())
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

def retrieve_with_hybrid(query: str, doc_ids: List[str], limit: int = 40) -> List[dict]:
    # Generate query variants
    query_variants = query_enhancer.generate_query_variants(query)
    
    # Multi-query vector search
    vector_results = qdrant_search_multi(query_variants, doc_ids, limit=limit)
    
    if USE_HYBRID_SEARCH and hybrid_searcher.bm25_index:
        return hybrid_searcher.hybrid_search(query, vector_results, doc_ids, limit)
    return vector_results[:limit]

# ---------- Rerank ----------
def rerank_candidates(original_query: str, cands: List[dict], top_n: int) -> List[dict]:
    if not cands:
        return []
    if not co:
        return cands[:top_n]

    texts = [c["metadata"]["text"] for c in cands]
    try:
        if len(texts) > 100:
            texts = texts[:100]
            cands_to_rerank = cands[:100]
        else:
            cands_to_rerank = cands

        rr = co.rerank(
            model="rerank-english-v3.0",
            query=original_query,
            documents=texts,
            top_n=min(top_n, len(texts)),
            return_documents=False
        )
        picked = []
        for r in rr.results:
            original = cands_to_rerank[r.index]
            original['rerank_score'] = float(r.relevance_score)
            picked.append(original)
        return picked
    except Exception as e:
        print(f"Cohere rerank failed: {e}, using original order")
        return cands[:top_n]

# ---------- Answer Generation ----------
def generate_concise_answer(question: str, context: str) -> str:
    prompt = f"""You are an expert analyst reviewing documents to answer questions.

Rules:
- Use ONLY facts present in the Context below
- If information is missing, say "Not specified in the provided context"
- Do not infer or use external knowledge
- Quote exact numbers, dates, percentages when present
- Keep answer to 1-3 sentences maximum
- No markdown or formatting

Question: {question}

Context:
{context}

Answer:"""
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": "Provide brief, accurate answers using only the provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error processing the question. Please try again."

def compress_if_needed(answer: str, question: str) -> str:
    # Remove any markdown formatting
    answer = re.sub(r'\*\*(.+?)\*\*', r'\1', answer)
    answer = re.sub(r'\*(.+?)\*', r'\1', answer)
    answer = re.sub(r'^#+\s+', '', answer, flags=re.MULTILINE)
    answer = re.sub(r'^[-*]\s+', '', answer, flags=re.MULTILINE)
    answer = re.sub(r'\|.+\|', '', answer)
    
    if len(answer) > 300:
        try:
            resp = client.chat.completions.create(
                model=GEN_MODEL,
                messages=[
                    {"role": "system", "content": "Extract only essential facts. Be extremely concise."},
                    {"role": "user", "content": f"Compress to <= 100 words, keep only essential facts.\n\nQ: {question}\nA: {answer}"},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            return resp.choices[0].message.content.strip()
        except:
            sentences = answer.split('. ')
            return '. '.join(sentences[:2]) + '.'
    return answer

# ---------- Main QA Pipeline ----------
def process_question(question: str, doc_ids: List[str]) -> str:
    print(f"\n📍 Q: {question[:80]}")

    # Retrieve with enhanced query
    results = retrieve_with_hybrid(question, doc_ids, limit=CANDIDATES_PER_QUERY)
    print(f"  Retrieved {len(results)} candidates")

    if not results:
        return "The requested information is not available in the provided documents."

    # Rerank if available
    print(f"  Reranking top {min(len(results), 100)}...")
    picked = rerank_candidates(question, results, top_n=RERANK_TOP_K)

    # Build context from top chunks (dedupe by text similarity)
    top = picked[:MAX_CTX_CHUNKS]
    context_parts = []
    seen_texts = set()
    
    for c in top:
        text = c["metadata"]["text"]
        # Simple deduplication by text hash
        h = hashlib.md5(text[:200].encode()).hexdigest()
        if h not in seen_texts:
            seen_texts.add(h)
            context_parts.append(text)

    context = "\n\n".join(context_parts)

    # Generate answer
    answer = generate_concise_answer(question, context)
    answer = compress_if_needed(answer, question)
    return answer

def process_questions_sequential(questions: List[str], doc_ids: List[str]) -> List[str]:
    answers = []
    print(f"\nProcessing {len(questions)} questions")
    print(f"Using: {'Hybrid (Vector + BM25)' if USE_HYBRID_SEARCH else 'Vector only'} | "
          f"Threshold: {SCORE_THRESHOLD} | Rerank: {'Cohere' if USE_RERANK else 'None'}")
    print(f"Pipeline: {CANDIDATES_PER_QUERY} candidates → Rerank {RERANK_TOP_K} → Context {MAX_CTX_CHUNKS}")

    start = time.time()
    for i, q in enumerate(questions):
        try:
            if i > 0:
                time.sleep(0.2)  # Rate limiting
            ans = process_question(q, doc_ids)
            answers.append(ans)
            print(f"✓ Q{i+1}: {len(ans)} chars")
        except Exception as e:
            print(f"✗ Q{i+1} failed: {e}")
            answers.append("Error processing the question. Please try again.")
    
    elapsed = time.time() - start
    print(f"\nAll questions processed in {elapsed:.1f}s ({elapsed/len(questions):.1f}s each)")
    return answers

# ---------- API Models ----------
class RunRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "healthy", "message": "Universal RAG System"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "config": {
            "mode": "DOMAIN_AGNOSTIC",
            "retrieval_pipeline": f"{CANDIDATES_PER_QUERY} → Rerank {RERANK_TOP_K} → {MAX_CTX_CHUNKS}",
            "chunk_size": "1200",
            "chunk_overlap": "400",
            "hybrid_search": USE_HYBRID_SEARCH,
            "bm25_index_loaded": hybrid_searcher.bm25_index is not None,
            "score_threshold": SCORE_THRESHOLD,
            "embed_model": EMBED_MODEL,
            "gen_model": GEN_MODEL,
            "use_rerank": USE_RERANK,
            "qdrant_collection": QDRANT_COLLECTION,
            "qdrant_connected": QDRANT_URL.lower() != "skip",
            "max_workers": MAX_WORKERS,
            "embedding_cache_size": len(embedding_cache),
        },
    }

@app.post("/run")
def run(payload: RunRequest, authorization: str = Header(None)):
    # Optional auth
    if HACKRX_API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing Bearer token")
        if authorization.split()[1] != HACKRX_API_KEY:
            raise HTTPException(403, "Invalid token")

    urls = payload.documents if isinstance(payload.documents, list) else [payload.documents]
    if not urls or not payload.questions:
        raise HTTPException(400, "documents and questions are required")

    try:
        total_start = time.time()
        print("\n" + "=" * 80)
        print(f"📝 Processing: {len(urls)} docs | {len(payload.questions)} questions")
        print(f"📊 Pipeline: {CANDIDATES_PER_QUERY} candidates → Rerank {RERANK_TOP_K} → Context {MAX_CTX_CHUNKS}")
        print("=" * 80)

        # Extract chunks
        t0 = time.time()
        chunks, metas = extract_chunks(urls)
        if not chunks:
            return {"answers": ["Document processing failed. No text could be extracted."] * len(payload.questions)}
        print(f"✓ Extraction: {time.time()-t0:.1f}s | {len(chunks)} chunks")

        # Get document IDs
        doc_ids = list({sha1(m["source"]) for m in metas})

        # Upsert to Qdrant + Build BM25
        t1 = time.time()
        upsert_chunks_to_qdrant(chunks, metas)
        print(f"✓ Indexing: {time.time()-t1:.1f}s")

        # Process questions
        t2 = time.time()
        answers = process_questions_sequential(payload.questions, doc_ids)
        print(f"✓ QA Processing: {time.time()-t2:.1f}s")

        ttl = time.time() - total_start
        avg_length = sum(len(a) for a in answers) / len(answers) if answers else 0

        print("=" * 80)
        print(f"✅ COMPLETED in {ttl:.1f}s")
        print(f"📏 Avg answer length: {avg_length:.0f} chars")
        print("=" * 80)

        if RETURN_FORMAT == "simple":
            return {"answers": answers}
        else:
            return {
                "answers": answers,
                "metadata": {
                    "mode": "DOMAIN_AGNOSTIC",
                    "chunks_processed": len(chunks),
                    "documents_processed": len(doc_ids),
                    "questions_answered": len(payload.questions),
                    "processing_time_seconds": round(ttl, 1),
                    "avg_answer_length": round(avg_length),
                    "use_hybrid_search": USE_HYBRID_SEARCH,
                    "use_rerank": USE_RERANK,
                    "pipeline": f"{CANDIDATES_PER_QUERY}→{RERANK_TOP_K}→{MAX_CTX_CHUNKS}",
                    "models": {
                        "embedding": EMBED_MODEL,
                        "generation": GEN_MODEL,
                    },
                },
            }
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"answers": ["Processing error. Please try again."] * len(payload.questions)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Universal RAG System")
    print("⚙️  Configuration:")
    print(f"  - Retrieval: {'Hybrid (Vector + BM25)' if USE_HYBRID_SEARCH else 'Vector only'}")
    print(f"  - Score Threshold: {SCORE_THRESHOLD}")
    print(f"  - Reranker: {'Cohere' if USE_RERANK else 'Disabled'}")
    print(f"  - Context: {MAX_CTX_CHUNKS} chunks | Rerank top {RERANK_TOP_K}")
    print(f"  - Models: {EMBED_MODEL} | {GEN_MODEL}")
    print(f"  - Domain: AGNOSTIC (works with any document type)")
    print(f"  - Qdrant: {QDRANT_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8000)