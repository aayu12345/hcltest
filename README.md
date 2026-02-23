# Customer Support AI Engine: Architecture & Data Flow

This document explains exactly how the **NLP Engine (TF-IDF)** and **Generative AI (Groq Llama 3)** work together to provide intelligent, 3-step customer support resolutions based on a historical ticket dataset.

---

## 🏗️ 1. High-Level Architecture Overview

The system uses a **Hybrid Search + Synthesis** approach:
1. **Traditional NLP (Fast & Accurate)**: Finds the top 3 most mathematically similar historical tickets to the user's new problem.
2. **Generative AI (Smart & Contextual)**: Reads those 3 historical resolutions and "synthesizes" them into 3 clean, highly actionable steps for the frontend UI.

---

## 🔍 2. Step 1: The Traditional NLP Engine (TF-IDF + Cosine Similarity)

Before the GenAI steps in, we must find the most relevant past resolutions. Sending the entire 30,000+ row CSV to an LLM is impossible (and expensive). Therefore, we use an NLP approach called **TF-IDF**.

### A. Data Preprocessing
- The system reads `customer_support_tickets.csv`.
- It isolates 5 key columns: `Ticket Type`, `Ticket Subject`, `Ticket Description`, `Resolution`, and `Ticket Priority`.
- It combines everything *except* the `Resolution` into a single "Feature Document" of text.
- It cleans the text (removes punctuation, converts to lowercase, removes double spaces).

### B. TF-IDF Vectorization (Model Training)
**TF-IDF** stands for *Term Frequency-Inverse Document Frequency*.
- The `TfidfVectorizer` goes through every combined ticket text in the dataset.
- It converts words into numbers (vectors), weighing them based on how important they are. 
  - *Example*: The word "the" appears everywhere, so it gets a low weight. The word "crashes" is rare and specific, so it gets a high weight.
- We save this mathematical matrix as `nlp_engine.pkl`.

### C. Matching the New Ticket (Cosine Similarity)
When a new ticket is submitted via the Frontend:
1. The new ticket data is vectorized using the exact same TF-IDF logic.
2. We run **Cosine Similarity**, which mathematically compares the angle between the *new ticket's vector* and *all 30,000 historical vectors*.
3. The system returns the **Top 3 Tickets** that have the highest similarity scores (closest to 1.0).
4. We extract the `Resolution` text from those top 3 historical matches.

---

## 🧠 3. Step 2: The Generative AI Engine (Groq / Llama 3)

Now we have 3 historical resolutions, but they might be messy, robotic, or contain old customer names. We need Generative AI to "clean" and "enhance" them.

### A. Constructing the Prompt Context
We take the Top 3 resolutions from the NLP engine, along with their Similarity Scores, and package them into a prompt string.

### B. Calling the Groq API (Llama-3.1-8b)
We send the strict prompt to the `Groq` API asking it to act as an expert customer support agent. 

**The Strict Rules given to the LLM:**
- Review the historical resolutions.
- Provide *exactly* 3 actionable suggestions for the user.
- **Do not** talk. Just output the numbered list.
- **Must** append the NLP `Similarity Score` to the end of each suggestion so the UI knows how accurate the data is.

### C. The Output Payload
Instead of pure TF-IDF returning awkward CSV text, the GenAI engine returns a perfectly formatted list of 3 actionable steps. Because the GenAI is strictly bound by the TF-IDF context, **it does not hallucinate false information**—it only synthesizes what the NLP engine mathematically proved was historically successful.

---

## 🚀 4. Usage Flow for Frontend Developers

1. **Instantiate**: `engine = RecommendationEngine()`
2. **Predict**: `result = engine.get_enhanced_recommendation(type, subject, description, priority)`
3. **Frontend Integration**: 
   The `result` dictionary outputs `enhanced_resolution`, containing the UI-ready text:
   
   ```text
   1. Update your graphics drivers to prevent standard rendering crashes. (Similarity Score: 0.8123)
   2. Verify the game cache files through the launcher to fix corrupted saves. (Similarity Score: 0.7451)
   3. Disable background overlay applications like Discord or GeForce. (Similarity Score: 0.6120)
   ```
   You can easily render `result['enhanced_resolution']` directly into your React/Next.js frontend.
