# Learn LangChain - Comprehensive Study Notes

A structured, beginner-friendly guide to mastering LangChain - the powerful Python framework for building LLM-powered applications. This repository contains detailed notes, examples, and best practices covering everything from foundational concepts to advanced patterns.

## Overview

This guide takes a systematic approach to learning LangChain, organized sequentially to build knowledge progressively. Whether you're building chatbots, RAG applications, or AI agents, these notes provide practical insights and implementation patterns.

## What's Covered

### Foundations
- **Models**: LLMs, Chat Models, Embedding Models
- **Model Parameters**: temperature, top_p, max_tokens, streaming, and more

### Input Engineering
- **Prompt Templates**: Dynamic single-turn and multi-turn prompts
- **Messages**: SystemMessage, HumanMessage, AIMessage
- **Chat Prompt Templates**: Multi-turn conversation handling
- **Message Placeholders**: Dynamic chat history injection
- **Prompt Engineering Techniques**: Chain-of-Thought, Tree of Thought, Few-shot, RAG

### Output Handling
- **Structured Output**: `with_structured_output()` method
- **Output Parsers**: Str, JSON, Structured, Pydantic parsers
- **Data Schemas**: TypedDict, Pydantic, JSON Schema comparison

### Building Blocks
- **Runnables**: Task-Specific Runnables and Runnable Primitives
- **LCEL (LangChain Expression Language)**: Pipe operator syntax and composition patterns
- **Chains**: Sequential, Parallel, and Conditional execution flows

### Coming Soon
- Document Loaders & Text Splitters
- Vector Stores & Retrievers
- RAG (Retrieval-Augmented Generation)
- Memory & State Management
- Agents & Tool Calling

## Key Features

- **Visual Diagrams**: ASCII flow diagrams for parallel and conditional chains
- **Comparison Tables**: Quick reference tables for choosing the right tools
- **Practical Examples**: Real-world code examples with detailed explanations
- **Best Practices**: Production-ready patterns and considerations
- **Progressive Learning**: Sequential structure that builds on previous concepts

## Repository Structure

```
learn-langchain/
├── README.md                    # This file
└── learn_langchain.md          # Complete study notes
```

## Learning Path

The notes are structured in a recommended learning order:

1. Start with **Models** to understand the foundation
2. Move to **Prompts** for input engineering
3. Learn **Structured Output** for handling responses
4. Master **Chains** and **Runnables** for building workflows
5. Explore **Knowledge Integration** (RAG, Vector Stores)
6. Implement **Memory** for stateful applications
7. Build **Agents** for advanced reasoning and tool use

## Quick Start

1. Clone this repository
2. Open `learn_langchain.md` for the complete guide
3. Follow the sequential learning roadmap
4. Practice with the provided code examples

## Credits & Acknowledgments

These notes are compiled from my personal learning journey through the **CampusX "Generative AI using LangChain"** playlist. Special thanks to CampusX for creating excellent educational content that made this learning path possible.

**Learning Resource**: [CampusX - Generative AI using LangChain Playlist](https://www.youtube.com/watch?v=pSVk-5WemQ0&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0)

## Use Cases

This guide will help you build:

- Conversational AI & Chatbots
- Question-Answering Systems over Documents (RAG)
- Code Generation & Analysis Tools
- Data Extraction & Summarization Pipelines
- Multi-step Reasoning Applications
- AI Agents with Tool Integration

## Contributing

Found a typo or have suggestions? Feel free to open an issue or submit a pull request!

## Connect

If you found these notes helpful, feel free to star the repository and share it with others learning LangChain!

---

**License**: MIT

**Note**: These are study notes intended for educational purposes. Always refer to the [official LangChain documentation](https://python.langchain.com/docs/get_started/introduction) for the most up-to-date information.