# Journal-Based LLM Agent: High-Level Design

## 1. Project Overview

**Goal:**  
Build an autonomous LLM agent capable of maintaining rich, structured conversations, extracting and classifying ideas, insights, and user preferences, and storing them in a persistent database. The system is designed to recreate the experience of a highly context-aware conversation partner ("Drew-clone") while supporting long-term memory, intent awareness, and context-driven reasoning.

**Motivation:**  
- Users naturally generate ideas in conversation rather than writing into static documents.  
- Traditional journaling tools lack feedback and cognitive scaffolding.  
- A journal-focused LLM agent can provide:  
  - Real-time context-aware dialogue  
  - Persistent memory of past sessions  
  - Thematic analysis and insight generation  
  - Structured retrieval for creative, reflective, or analytical purposes  

---

## 2. Key Objectives

1. **Persistent memory of conversations**  
   - Store atomic conversation fragments in a database (e.g., MongoDB)  
   - Apply tags, summaries, and metadata for future retrieval  

2. **Contextual reasoning**  
   - Reconstruct multi-turn context for coherent conversation  
   - Include relevant memory and insights without overwhelming the LLM  

3. **Intent and mode awareness**  
   - Detect conversation mode (design, exploration, reflection, evaluation)  
   - Shape retrieval and response based on mode  

4. **Insight extraction and user modeling**  
   - Derive user traits, preferences, and themes  
   - Track evolution of ideas over time  

5. **Retrieval-Augmented Generation (RAG)**  
   - Conditional retrieval of relevant memory or external documents  
   - Prioritize fragments by relevance, recency, and signal strength  

6. **Personality / style shaping**  
   - Establish a consistent reasoning and conversational style (“Drew effect”)  
   - Support feedback loops for adaptive responses  

---

## 3. Core Components

| Component | Function |
|-----------|---------|
| **Conversation Loop** | Manages user input, agent reasoning, and responses |
| **Context Builder** | Constructs LLM prompt by combining recent conversation, RAG fragments, user profile, and intent |
| **Intent Detection** | Classifies current user input into modes (design, exploration, reflection, evaluation) |
| **RAG / Retrieval** | Queries persistent memory or external sources for relevant fragments |
| **Fragment Storage** | Stores atomic conversation fragments with tags, summaries, and metadata |
| **Insights / Derived Memory** | Aggregates and interprets user behavior, preferences, and emerging themes |
| **User Profile** | Stores preferences, interaction style, and recurring themes |
| **Prompt Assembly** | Organizes system instructions, memory, recent context, and user input for LLM consumption |
| **Response Generation (LLM)** | Produces output using combined prompt and context layers |

---

## 4. Workflow Overview

1. **User Input** → captured by the Conversation Loop  
2. **Intent Detection** → classify mode of input  
3. **RAG Decision** → determine whether retrieval is needed  
4. **Retrieve & Filter** → fetch top-N relevant fragments  
5. **Context Assembly** → construct layered prompt:
   - System / personality instructions
   - Intent / mode
   - Relevant memory (RAG fragments)
   - Recent conversation
   - Current user input  
6. **LLM Response** → generate output  
7. **Fragment Storage** → classify and store new data in MongoDB  
8. **Insights / Profile Update** → update derived memory and user profile  

---

## 5. Data Buckets

1. **Persistent Memory**  
   - Session journaling (fragments + tags)  
   - Structured knowledge / extracted facts  

2. **Derived Memory**  
   - Aggregations, counts, trends  
   - Insights, themes, and cross-topic analysis  
   - User profile / preferences  

3. **Active Context (Per-Turn)**  
   - Recent conversation  
   - Relevant retrieved memory  
   - Intent / mode  

4. **External Augmentation (Optional)**  
   - Reference documents  
   - Domain-specific knowledge  

---

## 6. Implementation Notes

- Prioritize **relevance over completeness** when retrieving memory  
- Summarize aggressively if token limits are exceeded  
- Context assembly is **layered**, not just a dump of fragments  
- RAG and insights should be **intent-aware**  
- User feedback can be used for refinement but is **optional per turn**  

---

## 7. Key Benefits

- Recreates rich, continuous conversational experience  
- Persistent memory supports longitudinal reflection  
- Insights and derivations extend beyond raw conversation  
- Conditional RAG ensures context-driven, focused reasoning  
- Scales across multiple domains and conversation types  

---

## 8. Next Steps (Recommended)

1. Build a prototype **Context Builder**  
2. Implement **intent detection and mode classification**  
3. Integrate **RAG retrieval with MongoDB fragments**  
4. Define **fragment tagging and storage schema**  
5. Assemble **layered prompt for LLM**  
6. Evaluate response quality and refine context selection rule