# Learn Langchain

This guide provides a structured approach to learning LangChain - a powerful framework for building LLM-powered applications. The document is organized to take you from foundational concepts to advanced patterns, with practical examples and best practices throughout.

> special thanks to CampusX: This document compiles my learning note from their "Generative AI usinf LangChain playlist"

**Learning Path:** Start with the mental models of core components, then dive deep into each component sequentially. Each section builds on previous knowledge, so following the order is recommended for beginners.

**Sequential Learning Roadmap:**

- **Foundations**
  - Models (LLMs, Chat Models, Embedding Models)
  - Model Parameters (temperature, top_p, max_tokens, etc.)
- **Input Engineering**
  - Prompt Templates (PromptTemplate, ChatPromptTemplate)
  - Messages (SystemMessage, HumanMessage, AIMessage)
  - Message Placeholders
  - Prompt Engineering Techniques (CoT, ToT, Few-shot, RAG)
- **Output Handling**
  - Structured Output (`with_structured_output()`)
  - Output Parsers (Str, JSON, Structured, Pydantic)
  - Data Format Schemas (TypedDict, Pydantic, JSON Schema)
- **Building Blocks**
  - Runnables (Task-Specific & Primitives)
  - LCEL (LangChain Expression Language)
  - Chains (Sequential, Parallel, Conditional)
- **Knowledge Integration**
  - Document Loaders (PDF, CSV, Web, APIs)
  - Text Splitters (Character, Token, Recursive, Semantic)
  - Embedding Models & Vector Representations
  - Vector Stores (Chroma, Pinecone, FAISS, Weaviate)
  - Retrievers (Similarity, MMR, Contextual Compression)
  - RAG (Retrieval-Augmented Generation) Patterns
- **State Management**
  - Memory Types (Buffer, Window, Summary, Custom)
  - Conversation History Management
  - Context Window Optimization
- **Advanced Patterns**
  - Agents & Tool Calling
  - ReAct (Reasoning + Acting) Pattern
  - Multi-Agent Systems
  - Custom Tools & Function Calling
  - Streaming & Async Execution
  - Error Handling & Fallbacks

---

## Introduction

**What is LangChain?**

LangChain is a Python framework that simplifies building applications powered by Large Language Models (LLMs). It provides abstractions and tools to compose LLMs with external data sources, APIs, and custom logic into sophisticated AI applications.

**Why LangChain?**

- **Unified Interface**: Work with any LLM (OpenAI, Anthropic, Google, local models) through a consistent API
- **Composability**: Chain components together using intuitive syntax (LCEL)
- **Production-Ready**: Built-in support for streaming, async, error handling, and monitoring
- **Extensibility**: Easily integrate custom logic, tools, and data sources
- **Rich Ecosystem**: Pre-built integrations with vector stores, document loaders, and agent tools

**Core Philosophy:**

LangChain treats everything as composable "Runnables" - units of work that can be chained together using the pipe operator (`|`). This makes complex workflows readable and maintainable:

```python
chain = prompt_template | model | output_parser
result = chain.invoke({'input': 'your query'})
```

**Use Cases:**

- Chatbots & Virtual Assistants
- Question-Answering over Documents (RAG)
- Code Generation & Analysis
- Data Extraction & Summarization
- Multi-step Reasoning & Agents
- Content Generation Pipelines

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

---

### Prompts

**static vs dynamic prompt**: Static prompts are often inefficient and inflexible, as they require manual modification for each new scenario. Prompt templates (dynamic prompts/messages) address this by allowing dynamic insertion of values.

```
                           Model
                             |
                         [invoke]
                             |
              +--------------+--------------+
              |                             |
        single message                list of messages
   (single turn stand alone          (multi-turn conversation)
         queries)                             |
              |                     +---------+---------+
        +-----+-----+               |                   |
        |           |         Static Message      Dynamic Message
  Static Message    |      (SystemMessage,      (ChatPromptTemplate)
                    |       HumanMessage,
             Dynamic Message    AIMessage)
           (PromptTemplate)
```

follow this <a href="https://github.com/campusx-official/langchain-prompts/tree/main">repo</a> to understand code usage of:

- prompt template generator (dynamic single message) --> `prompt_generator.py`
- simple chaining of prompt template and model --> `prompt_ui.py`
- messages (list of message, static) --> `message.py`
- chat prompt template (list of message, dynamic) --> `chat_prompt_template.py`
- message placeholder (list of message, dynamic) --> `message_placeholder.py`

#### Prompt Templates | Dynamic Message (Single-turn)

**Core Template Structure:**

- **Pre-written Template String:** Use case-specific prompt string that serves as the base template.
- **Input Variables:** Define placeholders (e.g., `{query}`, `{context}`) within your prompt string using curly braces. These are the dynamic parts.
- **Invoking Variables:** Pass a dictionary to the `PromptTemplate.format()` method with keys matching your input variables.

**Standard Template Components:**

- **Role Definition:** Assign a clear persona or role to the LLM (e.g., "You are a helpful assistant...", "You are an expert Python programmer...") to influence its tone, style, and the type of information it provides.
- **Context Section:** Provide sufficient and relevant background information for the LLM to generate accurate and helpful responses (user preferences, domain knowledge, previous conversation turns).
- **Task Instruction:** Clearly define what you want the LLM to do. Be specific and unambiguous about the expected behavior.
- **Output Format Specification:** Explicitly define the desired output format (e.g., JSON, XML, bullet points, markdown table) to ensure structured and parseable responses.
- **Constraints & Guardrails:** Set clear limits on the LLM's response (maximum length, tone requirements, content restrictions, safety instructions).

**Template Enhancement Techniques:**

- **Few-shot Examples:** Embed high-quality input-output examples directly into the template to guide the LLM's behavior. Particularly useful for:
  - Complex or nuanced tasks
  - Specific formatting requirements
  - Demonstrating desired reasoning patterns
- **Variable Placeholders:** Use clear, descriptive variable names (e.g., `{user_query}`, `{document_context}`, `{max_words}`) for better template readability and maintenance.
- **Input Variable Validation:** use `validate_template=True` in the template object for extra layer of validating the input variables.

#### Best Practices & Considerations

**Clarity & Precision:**

- Be specific, not vague; ambiguity → unpredictable results
- Break complex instructions into numbered/bulleted steps

**Context Management:**

- Provide necessary context only (general → specific structure)
- Respect token limits; balance detail with token budget

**Output Control:**

- Always specify output format explicitly
- Include examples for complex formats
- Define length, style, and content constraints

**Token Efficiency:**

- Design concise and efficient templates
- Be mindful of underlying LLM token limits, especially when incorporating examples

**Maintenance & Reusability:**

- Serialize templates (JSON/YAML) for version control and sharing
- Use descriptive names; document purpose and behavior

**Safety & Quality:**

- Add guardrails against harmful/biased/off-topic content
- Test with edge cases and adversarial inputs

**Iterative Development & Monitoring:**

- Start simple → test → analyze → refine
- Track modifications; small changes = significant impact
- A/B test template variations when possible
- Monitor for degradation when models are updated

**Domain Adaptation:**

- Customize templates for specific domains (technical, creative, conversational)
- Adjust formality, terminology, and structure based on target audience
- Consider cultural and linguistic nuances when applicable

#### Prompt Engineering Techniques

**Reasoning Enhancement Techniques:**

- **Chain-of-Thought (CoT) Prompting:** Structure prompts to encourage step-by-step reasoning. Include examples of intermediate thought processes to guide the LLM's reasoning pattern.
- **Tree of Thought (ToT):** Enable the LLM to explore multiple reasoning paths systematically. Structures prompts for complex problem-solving with branching exploration of solution spaces.
- **Self-Consistency:** Generate multiple CoT reasoning paths from a single prompt and aggregate results. Reduces reliance on a single LLM output for more robust answers.

**Knowledge Augmentation Techniques:**

- **Generated Knowledge Prompting:** Design prompts that first instruct the LLM to generate relevant background knowledge, then use that knowledge to answer the primary query. Enhances accuracy for knowledge-intensive tasks.
- **Retrieval-Augmented Generation (RAG):** Combine prompt templates with external knowledge retrieval. Template structures the query for both retrieval and generation phases.

**Agent-Based Techniques:**

- **ReAct (Reasoning + Acting):** For agentic workflows, combine explicit reasoning steps with tool-use actions. Template guides the LLM through: Reason → Act → Observe → Reason cycle, enabling dynamic interaction with external environments.

**Meta-Techniques:**

- **Automatic Prompt Engineer (APE):** Automated iterative refinement using an LLM to generate and evaluate candidate prompt instructions. Systematically improves template effectiveness without constant manual trial-and-error.

#### Messages

there are 3 boiled down messages in a chat:

- system message
- human message
- AI message

use these to maintain chat history --> **data annotation/labeled message**: which message coming from whom with additional metadata

> [applicable for prompt template, not chat prompt template] use `SystemMessage`, `HumanMessage` abd `AIMessage` from `langchain_core.messagees`

#### Chat Prompt Templates | Dynamic Message (Multi-turn Conversation)

This is a prompt template but for multi turn coversation. it may consists of system, human and AI messages and can be made dynamic.

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})
```

#### Message Placeholder

A `MessagesPlaceholder` in langchain is a special placeholder used inside a `ChatPromptTemplate` to dynamically insert chat history or a list of messages at runtime.

**Why Message Placeholder when we have Chat Prompt Template?**

`ChatPromptTemplate` allows you to define static message structures with dynamic values (e.g., `{topic}`, `{domain}`), but it doesn't handle **variable-length message lists**. `MessagesPlaceholder` solves this by:

- **Dynamic Chat History:** Insert entire conversation histories of varying lengths (could be 2 messages or 200 messages)
- **Flexible Message Lists:** Handle scenarios where the number of messages isn't known at template design time
- **Runtime Message Injection:** Pass complete message objects (with roles and content) rather than just string values

Without `MessagesPlaceholder`, you'd need to hardcode the exact number of messages in your template, which is impractical for conversational applications with growing chat histories.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'), # --> we can parse any variable value here in runtime
    ('human','{query}')
])

chat_history = []
# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})
```

---

### Structured Output

have language models return responses in a **well-defined data format** (e.g. json) rather than free-form text. This makes the model output easier to parse and work with programmatically.

two types of models:

- model that can give structured output (use: `with_structured_output()`)
- model that can't give structured output (use: Output Parser)

#### with_structured_output Function

```python
from typing, import TypedDict, Annotated

class DataFormat(TypedDict): # can use pydantic basemodel
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]

structured_model = model.with_structured_output(DataFormat) # this generates a system prompt that descrives the format to the LLM
# example system prompt behind the scene 9handled by the `with_structured_output()` method
"""
you are an AI assistant that extraects structured insights from text. Given a product review, extract:
    - summary: a brief overview of the main points
    - sentiment: overall tone of the review (postiive, negative, neutral)
Return the response in json format.
"""

result = structured_model.invoke(prompt)

print(result) # {'summary': ..., 'sentiment': ...}
print(result['summary']) # ...
print(result['sentiment']) # ...
```

> just define and mention the data format you want.

[note] in `with_structured_output()` method explore these parameters:

- `json_mode` --> want output as json format
- `function_calling` --> call another function with the output

3 ways to specify the data format:

- TypedDict: Basic structure and type enforcement only
- Pydantic: All features except cross-language compatibility (best for Python-centric apps with validation needs)
- JSON Schema: Basic structure, type enforcement, validation, and cross-language compatibility (but no default values or automatic conversion)

when to use what:

| Feature                      | TypedDict | Pydantic | JSON Schema |
| ---------------------------- | --------- | -------- | ----------- |
| Basic structure              | ✔         | ✔        | ✔           |
| Type enforcement             | ✔         | ✔        | ✔           |
| Data validation              | x         | ✔        | ✔           |
| Default values               | x         | ✔        | x           |
| Automatic conversion         | x         | ✔        | x           |
| Cross-language compatibility | x         | x        | ✔           |

#### Output Parsers

Output Parsers in langchain help **converting raw LLM responses into structured formats** like json, csv, pydantic models and more. They ensure consistency, validation and ease of use in apps.

There are many output parsers in langchain. most commonly used are:

- `StrOutputParser` - Extracts raw string output from LLM responses
- `JsonOutputParser` - Parses LLM output into valid JSON objects but LLM decides the json schema
- `StructuredOutputParser` - Converts LLM responses into predefined structured formats (validation not enforced) with field definitions
- `PydanticOutputParser` - Validates and parses LLM output into type-safe Pydantic models

when to use what:

| Feature                     | StrOutputParser | JsonOutputParser | StructuredOutputParser | PydanticOutputParser |
| --------------------------- | --------------- | ---------------- | ---------------------- | -------------------- |
| Raw string output           | ✔               | x                | x                      | x                    |
| JSON parsing                | x               | ✔                | ✔                      | ✔                    |
| Schema definition           | x               | x                | ✔                      | ✔                    |
| Type validation             | x               | x                | x                      | ✔                    |
| Field descriptions          | x               | x                | ✔                      | ✔                    |
| Default values              | x               | x                | x                      | ✔                    |
| Automatic type conversion   | x               | x                | x                      | ✔                    |
| Error handling & validation | x               | x                | x                      | ✔                    |

**why use parser?**
It plays very well with chain component and reduce code complexity.

Example usage simple yet powerful `StrOutputParser` or any parser that matters:

```text
[example workflow to implement] topic --> LLM --> detailed report --> LLM --> 5 line summary
```

```python
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'black hole'})
```

for more visit <a href="https://github.com/campusx-official/langchain-output-parsers">this repo</a>

---

### Chains

yep, make simple or complex pipelines for your whole application!

- `sequential`
- `parallel`
- `conditional`
- `complex`

> use `chain.get_graph().print_ascii()` to visualize the pipeline flow

**Parallel Chain Example:**

```text
Flow diagram:
                           ┌─> prompt1 -> model -> parser1 ─┐
{inputs} -> parallel_step ─┤                                ├─> prompt3 -> model -> parser3 -> result
                           └─> prompt2 -> model -> parser2 ─┘
```

```python
from langchain.schema.runnable import RunnableParallel

# Step 1: Two chains execute in parallel with different inputs
parallel_step = RunnableParallel(
    product_info=prompt1 | model | parser1,        # parallel chain 1: uses {product_name}
    customer_history=prompt2 | model | parser2     # parallel chain 2: uses {customer_id}
)

# Step 2: Combine parallel results and feed to final chain
final_chain = parallel_step | prompt3 | model | parser3

# Full execution: parallel chains -> sequential final chain
result = final_chain.invoke({
    'product_name': 'iPhone 15',
    'customer_id': 'C12345'
})
# parallel_step output: {'product_info': '...', 'customer_history': '...'}
# final output: personalized recommendation based on both inputs
```

**Conditional Chain Example:**

```text
Flow diagram:
                    ┌─> condition1 == True  -> chain1 ─┐
                    │                                   │
{input} -> branch_chain ─┼─> condition2 == True  -> chain2 ─┼─> result
                    │                                   │
                    └─> else (default)     -> default_chain ─┘
```

```python
from langchain.schema.runnable import RunnableBranch, RunnableLambda

branch_chain = RunnableBranch(
    (condition1, chain1),     # branch 1: if condition1 is True
    (condition2, chain2),     # branch 2: if condition2 is True
    default_chain             # default: if no condition is met (use RunnableLambda for dummy chain)
)
```

for more code examples go through <a href="https://github.com/campusx-official/langchain-chains">this repo</a>

---

#### Runnables

Runnables are the fundamental building blocks of LangChain chains - they represent **units of work** that can be composed together.

**Core Concept:**

- **Input → Process → Output**: Every runnable takes an input, processes it, and produces an output
- **Standardized Interface**: All runnables implement the same interface (`invoke()`, `batch()`, `stream()`, `ainvoke()`)
- **Composable**: Can be connected using the pipe operator (`|`) to build chains
- **Type-Safe**: Input/output types are clearly defined for better error handling

**Key Characteristics:**

- **Chainable**: Connect multiple runnables sequentially: `runnable1 | runnable2 | runnable3`
- **Parallelizable**: Execute multiple runnables simultaneously using `RunnableParallel`
- **Conditional**: Branch execution based on conditions using `RunnableBranch`
- **Reusable**: Same runnable can be used in multiple chains
- **Debuggable**: Built-in support for tracing and visualization (`get_graph()`)

**Standard Methods:**

- `invoke(input)`: Synchronous single execution
- `batch([inputs])`: Process multiple inputs at once
- `stream(input)`: Stream output incrementally
- `ainvoke(input)`: Async single execution (for concurrent operations)

**Why Runnables?**

- **Consistency**: Same interface across all components (prompts, models, parsers, etc.)
- **Flexibility**: Easy to swap components without changing chain structure
- **Performance**: Built-in optimization for parallel and streaming execution
- **Maintainability**: Clear data flow and modular architecture

##### Common Runnable Types

Runnables can be divided in two categories:

**1. Task-Specific Runnables**

- Definition: These are core LangChain components that have been converted into Runnables so they can be used in pipelines.
- Purpose: Perform task-specific operations like LLM calls, prompting, retrieval, etc.

**Examples:**

- `ChatOpenAI` → Runs an LLM model
- `PromptTemplate` → Formats prompts dynamically
- `Retriever` → Retrieves relevant documents
- `OutputParser` → Parses and structures model outputs

**2. Runnable Primitives**

- Definition: These are fundamental building blocks for structuring execution flow.
- Purpose: They help orchestrate execution by defining how different Runnables are combined (sequentially, in parallel, conditionally, etc.).

**Examples:**

- `RunnableSequence` → Runs steps in order (`|` operator)
- `RunnableParallel` → Runs multiple steps simultaneously
- `RunnableMap` → Maps the same input across multiple functions
- `RunnableBranch` → Implements conditional execution (if-else logic)
- `RunnableLambda` → Wraps custom Python functions into Runnables
- `RunnablePassthrough` → Just forwards input as output (acts as a placeholder)

#### Langchain Expression Language (LCEL)

LCEL is LangChain's declarative syntax for building chains using the pipe operator (`|`). It's a simple, intuitive way to compose Runnables into complex workflows.

**What is LCEL?**

LCEL is the syntax/interface that allows you to chain Runnables together using `|`. Think of it as the "glue" that connects components.

```python
# LCEL syntax: component1 | component2 | component3
chain = prompt_template | model | output_parser
result = chain.invoke({'input': 'value'})
```

**Key Benefits:**

- **Readability**: Chains read left-to-right like Unix pipes (`data | transform | process`)
- **Simplicity**: No need for complex class inheritance or boilerplate code
- **Composability**: Mix and match any Runnables (prompts, models, parsers, custom functions)
- **Automatic Features**: Get streaming, async, batching, and parallel execution for free

**LCEL vs Traditional Approach:**

```python
# Traditional approach (verbose)
prompt = prompt_template.format(topic="AI")
response = model.invoke(prompt)
parsed = output_parser.parse(response)

# LCEL approach (concise)
chain = prompt_template | model | output_parser
result = chain.invoke({'topic': 'AI'})
```

**Common LCEL Patterns:**

```python
# Sequential: A -> B -> C
chain = step1 | step2 | step3

# Parallel: (A, B) -> C
parallel = RunnableParallel(a=chain1, b=chain2)
full_chain = parallel | step3

# Conditional: if-else branching
branch = RunnableBranch((condition, chain1), chain2)

# With custom functions
chain = prompt | model | RunnableLambda(custom_function) | parser
```

**Why LCEL Matters:**

LCEL is the modern way to build LangChain applications. It replaces older patterns (like `LLMChain`, `SequentialChain`) with a unified, more powerful approach that works seamlessly with all Runnables.
