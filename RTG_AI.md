# Project-Based LLM Learning Path

A hands-on roadmap for learning LLMs, agents, and RAG through practical projects.

---

## Phase 1: Build Your First LLM App (Week 1-2)

**Goal:** Get something working immediately  
**Tools:** `transformers`, `langchain`  
**Project:** Build a chatbot that can have a conversation with context memory  
**What you'll learn:** Basic LLM interaction, prompting, conversation handling

---

## Phase 2: RAG System from Scratch (Week 3-4)

**Goal:** Build a "chat with your docs" system  
**Tools:** `langchain`, `chromadb` or `faiss`, `sentence-transformers`  
**Project:** PDF/document Q&A system that retrieves relevant chunks  
**What you'll learn:** Embeddings, vector databases, retrieval, chunking strategies

---

## Phase 3: Function Calling & Tool Use (Week 5-6)

**Goal:** LLM that can use external tools  
**Tools:** `langchain` (tools & agents), OpenAI/Anthropic APIs  
**Project:** Build an LLM assistant that can search web, do calculations, query APIs  
**What you'll learn:** Tool definitions, structured outputs, error handling

---

## Phase 4: MCP Integration (Week 7)

**Goal:** Connect LLM to real data sources  
**Tools:** `mcp` (Model Context Protocol), `langchain`  
**Project:** LLM that can access your Google Drive, GitHub, databases via MCP servers  
**What you'll learn:** MCP servers/clients, resource/tool protocols, real integrations

---

## Phase 5: Simple Agent with LangGraph (Week 8-9)

**Goal:** Build a stateful agent that can plan and execute  
**Tools:** `langgraph`, `langchain`  
**Project:** Research assistant that can break down queries, search, synthesize findings  
**What you'll learn:** Agent loops, state management, ReAct pattern, control flow

---

## Phase 6: Advanced RAG (Week 10-11)

**Goal:** Production-quality retrieval  
**Tools:** `langchain`, `llama-index`, rerankers  
**Project:** Upgrade Phase 2 with query rewriting, HyDE, reranking, multi-index  
**What you'll learn:** Advanced retrieval patterns, hybrid search, metadata filtering

---

## Phase 7: Fine-tuning & Customization (Week 12-13)

**Goal:** Customize a model for your needs  
**Tools:** `transformers`, `peft` (LoRA), `trl`  
**Project:** Fine-tune a 7B model on domain-specific data with LoRA  
**What you'll learn:** Training loop, LoRA adapters, dataset preparation, evaluation

---

## Phase 8: Multi-Agent System (Week 14-15)

**Goal:** Agents that work together  
**Tools:** `langgraph`, `langchain`  
**Project:** Build a team of agents (researcher, writer, critic) that collaborate  
**What you'll learn:** Agent communication, delegation, consensus, orchestration

---

## Phase 9: Local LLM Setup (Week 16)

**Goal:** Run everything locally  
**Tools:** `ollama`, `vllm`, `llama.cpp`  
**Project:** Set up local inference server, integrate with your existing apps  
**What you'll learn:** Quantization, inference optimization, serving, privacy

---

## Phase 10: Production-Ready System (Week 17-18)

**Goal:** Deploy something real  
**Tools:** `langsmith` or `langfuse` (observability), `fastapi`  
**Project:** Deploy one of your previous projects with monitoring, caching, error handling  
**What you'll learn:** Tracing, evaluation, cost optimization, production concerns

---

## Tool Timeline at a Glance

| When                  | Tools to Learn                             |
| --------------------- | ------------------------------------------ |
| **Start immediately** | `transformers`, `langchain`                |
| **Week 3**            | Vector DBs (`chromadb`, `faiss`)           |
| **Week 7**            | `mcp` (Model Context Protocol)             |
| **Week 8**            | `langgraph` (for complex agents)           |
| **Week 10**           | `llama-index` (alternative RAG framework)  |
| **Week 12**           | `peft`, `trl` (fine-tuning)                |
| **Week 16**           | `ollama`, `vllm` (local/efficient serving) |
| **Week 17**           | `langsmith`/`langfuse` (observability)     |

---

## Key Points

✅ **RAG comes early** (Phase 2) - it's foundational and you'll keep improving it  
✅ **MCP comes after** you understand tools (Phase 4) - it's about connecting things  
✅ **LangGraph comes mid-way** (Phase 5) - when simple chains aren't enough  
✅ **Each project builds on previous** - you'll keep using earlier code  
✅ **Theory emerges naturally** - you learn what you need when you need it

---

## Next Steps

When you're ready to start any phase, discuss it for detailed breakdown with:

- Code structure
- Step-by-step implementation
- Resources and documentation
- Common pitfalls to avoid

phase planning template:

```text
Plan Phase X: [Phase Name]

Give me:
1. Goal breakdown
2. Theory/concepts to understand
3. Hands-on implementation plan
```
