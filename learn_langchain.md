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
