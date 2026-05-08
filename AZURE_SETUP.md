# Azure Portal Setup Guide
## CV to JD Mapping System v2

This guide walks you through creating all required Azure resources.
Follow these steps in order.

---

## 1. Create Azure OpenAI Resource (LLM Endpoint)

1. Go to [portal.azure.com](https://portal.azure.com) → **Create a resource**
2. Search for **"Azure OpenAI"** → Click **Create**
3. Fill in:
   - **Subscription**: your subscription
   - **Resource Group**: create new → `rg-cv-jd-mapper` (or use existing)
   - **Region**: `East US` or `Sweden Central` (best model availability)
   - **Name**: `cv-jd-openai` (must be globally unique)
   - **Pricing tier**: Standard S0
4. Click **Review + Create** → **Create**
5. Once deployed, go to **Resource** → **Keys and Endpoint**
   - Copy **Endpoint** (e.g. `https://cv-jd-openai.openai.azure.com/`)
   - Copy **Key 1**

### Deploy Models

In your Azure OpenAI resource → **Model deployments** → **Manage Deployments** → **Azure OpenAI Studio**:

#### LLM Deployment (for analysis)
- Click **+ New deployment**
- Model: `gpt-4.1-mini` (or `gpt-4o-mini` if 4.1 not available)
- Deployment name: `gpt-4.1-mini`  ← use this exact name in .env
- Click **Deploy**

#### Embedding Deployment (for FAISS)
- Click **+ New deployment**
- Model: `text-embedding-ada-002`  (cheaper; 1536-dim vectors)
- Deployment name: `text-embedding-ada-002`  ← use this exact name in .env
- Click **Deploy**

### Update .env

```env
AZURE_OPENAI_ENDPOINT=https://cv-jd-openai.openai.azure.com/
AZURE_OPENAI_KEY=<Key 1 from portal>
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_API_VERSION=2023-05-15
```

---

## 2. Create Azure Storage Account (Blob Storage)

1. Go to **Create a resource** → search **"Storage account"** → Click **Create**
2. Fill in:
   - **Resource Group**: `rg-cv-jd-mapper` (same as above)
   - **Storage account name**: `cvjdmapperstorage` (must be globally unique, 3-24 chars, lowercase)
   - **Region**: same as OpenAI resource
   - **Performance**: Standard
   - **Redundancy**: LRS (Locally Redundant — cheapest)
3. Click **Review + Create** → **Create**

### Get Connection String

In your Storage account → **Security + networking** → **Access keys**:
- Click **Show keys**
- Copy **Connection string** (under key1)

### Create Blob Container

In your Storage account → **Data storage** → **Containers** → **+ Container**:
- Name: `cv-jd-index`
- Access level: **Private**
- Click **Create**

### Update .env

```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=cvjdmapperstorage;AccountKey=<your-key>;EndpointSuffix=core.windows.net
AZURE_STORAGE_CONTAINER=cv-jd-index
```

---

## 3. Verify Setup

Run this quick test in Python (from project root):

```python
from config.settings import settings, get_llm_client, get_embeddings_client
from utils.vector_store import FAISSJDIndex

# Test LLM connection
client = get_llm_client()
resp = client.chat.completions.create(
    model=settings.azure_openai_deployment,
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=5
)
print("LLM:", resp.choices[0].message.content)

# Test embeddings
emb = get_embeddings_client()
vec = emb.embed_query("test")
print("Embedding dim:", len(vec))   # Should be 1536 for text-embedding-ada-002

# Test blob storage
index = FAISSJDIndex()
index.save_to_blob()
print("Blob storage: OK")
```

---

## 4. UltraTech Corporate Network (SSL)

If running inside UltraTech's network, you may need the corporate SSL cert:

1. Export the cert from your browser or ask IT for the `.pem` file
2. Set in .env:
   ```env
   SSL_CERT_PATH=/path/to/ultratech-corp-cert.pem
   ```
3. The `get_llm_client()` and `get_embeddings_client()` factories automatically use this cert
   when `SSL_CERT_PATH` is set.

---

## Resource Summary

| Resource | Name | Used For |
|----------|------|----------|
| Azure OpenAI | `cv-jd-openai` | GPT-4.1-mini (analysis) + text-embedding-ada-002 |
| Azure Storage | `cvjdmapperstorage` | FAISS index, CV files, JD files (blob storage) |
| Container | `cv-jd-index` | All blob data |
