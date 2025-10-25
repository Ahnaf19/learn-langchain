# Learn Langchain

```python
# TODO: add learning introduction
```

---

## Introduction

```python
# TODO: add langchain introduction
```

---

## Components

6 major components:

1. models
2. prompts
3. chains
4. memory
5. indexes
6. agents

### Short Mental-models of the Compoments

#### Models

in langchain, "models" are the core interfaces through which you interact with AI models.

> abstracts away the difference between different llm apis like openai, anthropic by giving a single interface

2 types of models:

- chat models (https://python.langchain.com/docs/integrations/chat/)
- embedding models (https://python.langchain.com/docs/integrations/text_embedding/)

#### Prompts

input to llms

- `dynamic & reusable` prompts
- `role-based` prompts (system level and user level prompts)
- `few shot` prompting (example based)

#### Chains

**pipelines**

- `sequential`
- `parallel`
- `conditional`
- `complex`

#### Indexes

Indexes connecyour application to external knowledge such as pdfs, websites or databases

4 things in indexes:

1. doc loader
2. text splitter
3. vectore store
4. retrivers

#### Memory

LLM API calls are **stateless**

- `ConversationBufferMemory`: stores a transcript of recent messages. great for short chats but can grow large quickly
- `ConversationBufferWindowMemory`: only keeps the lat N interactions to avoid excessive token usage
- `Summarizer-Based Memory`: peridically summarizes older chat segmentsto keep a condensed memory footprint.
- `Custom Memory`: for advanced use cases, you can store specialied state (the user's preferences or key facts about them) in a custom meory class

#### Agents

LLM ==> NLU + TEXT GEN

example: chatbot --> "if llm can understand and generate text based on that, it can do some work for too right?" --> the think behind AI Agent

> so, AI Agent ==> chatbot + super power

these super powers are:

- reasoning capabilities (for example technique to breakdown prompt/query: chain of thought)
- tools

---

## Deep Dive

**temperature**: A float value (0-2) controlling randomness/creativity. Lower values (0-0.3) for deterministic output; higher values (0.7-1.0) for creative output.

**max_tokens** / **max_output_tokens**: An integer specifying the maximum number of tokens the model can generate. Controls response length and API costs.

**top_p** (nucleus sampling): A float (0-1) alternative to temperature. Considers tokens whose cumulative probability mass equals top_p. Lower values for more focused output.

**frequency_penalty**: A float (-2.0 to 2.0) reducing likelihood of repeating tokens based on frequency. Positive values discourage repetition.

**presence_penalty**: A float (-2.0 to 2.0) reducing likelihood of repeating any token. Positive values encourage new topics.

**n**: An integer (1-10) for the number of chat completion choices to generate. Returns multiple alternative responses.

**stop**: A string or list of strings (up to 4) where the model will stop generating tokens.

**streaming**: A boolean (True/False). When True, responses are returned incrementally.

**model_name** / **model**: A string identifier specifying the model variant to use.

**timeout**: An integer or float (seconds) for the maximum time to wait for an API response.

**request_timeout**: An integer (seconds) for the maximum duration for the HTTP request to complete.

**seed**: An integer that makes outputs more reproducible.

**top_k**: An integer limiting the model to considering only the k most likely next tokens.

**logit_bias** / **logprobs**: A dictionary mapping token IDs to bias values (-100 to 100). Modifies the likelihood of specific tokens appearing.

**response_format**: A dictionary specifying format type (e.g., {"type": "json_object"}). Constrains model output to specific formats.

Usage Tips:

- Start with **temperature=0.7** for balanced responses
- Use **temperature=0** for deterministic/factual tasks
- Combine **max_tokens** with cost monitoring
- **top_p** and **temperature** shouldn't typically be modified together (use one or the other)

### Models

The Model Component in LangChain is a crucial part of the framework, designed to facilitate
interactions with various language models and embedding models.

It abstracts the complexity of working directly with different LLMs, chat models, and
embedding models, providing a **uniform interface to communicate** with them. This makes it
easier to build applications that rely on AI-generated text, text embeddings for similarity
search, and retrieval-augmented generation (RAG).

models:

- language models
  - LLMs [string input --> string output] (general purpose)
  - Chat Models [string input --> chat message output] (conversational)
- embedding models [text input --> numerical vector output] (semantic representation for similarity, search, RAG)
  - key for RAG: uses cosine similarity for semantic search in knowledge base

| Feature          | LLMs (Base Models)                                                             | Chat Models (Instruction-Tuned)                                              |
| ---------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Purpose          | Free-form text generation                                                      | Optimized for multi-turn conversations                                       |
| Training Data    | General text corpora (books, articles)                                         | Fine-tuned on chat datasets (dialogues, user-assistant conversations)        |
| Memory & Context | No built-in memory                                                             | Supports structured conversation history                                     |
| Role Awareness   | No understanding of "user" and "assistant" roles                               | Understands "system", "user", and "assistant" roles                          |
| Example Models   | GPT-3, Llama-2-7B, Mistral-7B, OPT-1.3B                                        | GPT-4, GPT-3.5-turbo, Llama-2-Chat, Mistral-Instruct, Claude                 |
| Use Cases        | Text generation, summarization, translation, creative writing, code generation | Conversational AI, chatbots, virtual assistants, customer support, AI tutors |
