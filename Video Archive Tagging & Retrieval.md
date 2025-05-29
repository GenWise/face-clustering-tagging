# 🎬 AI Strategy for Video Archive Tagging & Retrieval

This document summarizes our discussion on building an AI-enhanced semantic indexing and retrieval system for various types of video content collected by GenWise.

## 🧱 Vision

Enable a small internal team to:

* Search, filter, and retrieve relevant video clips using natural language.
* Automatically tag key people, activities, and settings using LLMs, transcripts, and computer vision.
* Maintain a structured CSV-based metadata layer across all videos to drive search and chatbot functionality.

---

## 🗂️ Video Types & AI Opportunities

### 1. **Summer Camp Activity Videos**

**Format:** 5s–3m, 1000s/year, \~4 years of archives

**Current Pipeline:**

* ✅ Face recognition complete (known individuals per video)
* ✅ Keyframe extraction implemented

**Next Steps:**

* Whisper transcription (if audio exists)
* LLM-based activity, setting, gender-mix tagging via GPT-4o or Claude
* Output structured CSV/JSON per video

**Applications:**

* Retrieve clips like *"Girls from School A doing STEM activities outdoors"*
* Curate thematic reels for events, marketing, archives

---

### 2. **Online Teacher Courses**

**Format:** 40–50 sessions/year, 90–120 min each

**Available Inputs:**

* Zoom transcripts
* Assignments, questions, feedback

**Next Steps:**

* Chunk transcripts and summarize by concept/session
* Extract mentor insights, recurring questions, and instructional patterns
* Build private search/chatbot interface over all sessions

**Applications:**

* Mentor support
* Institutional memory
* Internal training for new instructors

---

### 3. **Interviews with School Leadership**

**Format:** 30–60 min; \~100 interviews

**Next Steps:**

* Transcript segmentation
* Speaker turn detection
* Extract key quotes, themes (pedagogy, leadership challenges, innovations)
* Search by theme/person/school

**Applications:**

* Showcase best practices
* Compare pedagogical priorities
* Inform strategic decisions

---

### 4. **Online Classes with Students**

**Format:** \~150+ hours/year

**Next Steps:**

* Summarize transcripts (key questions, answers, moments of insight)
* Classify instructional strategies used
* Less emphasis on face recognition due to Zoom limitations

**Applications:**

* Highlight reels of student curiosity or breakthroughs
* Curriculum mapping
* Teacher training

---

## 🔍 AI Modalities to Integrate

| Capability                  | Applies To | Tools/Models                        |
| --------------------------- | ---------- | ----------------------------------- |
| Face recognition            | 1          | Existing pipeline                   |
| Whisper transcription       | 1, 4       | Whisper API or local                |
| Transcript chunking + Q\&A  | 2, 3, 4    | Claude, GPT-4o                      |
| Activity detection via LLMs | 1          | GPT-4o with images                  |
| Long video summarization    | 2, 3, 4    | Claude 3 Opus, GPT-4-turbo          |
| Insight/quote extraction    | 2, 3       | LLMs + regex                        |
| CSV-based filtering         | All        | Pandas/SQLite                       |
| Semantic search/chatbot     | All        | LangChain, Claude Tools, GPT w/ CSV |

---

## 🧠 Strategic Directions

1. **CSV as Semantic Backbone:**

   * Standardize fields like `video_id`, `people_present`, `activity`, `setting`, `gender_mix`, `schools`, `transcript_excerpt`, `tags`, etc.

2. **Prompting Strategy:**

   * Start with **unified prompts** to extract all tags at once
   * Modularize later if needed

3. **Chatbot for Retrieval:**

   * Feasible to create an AI chatbot over the CSV
   * Use GPT-4 Code Interpreter, LangChain, or Claude Tools for semantic querying

4. **Frontend Review (Optional Later):**

   * Flask/Streamlit UI to review and approve tags
   * Allows human-in-the-loop feedback to improve quality

---

## 🛣️ Suggested Roadmap

**Phase 1:**

* Whisper integration on summer camp videos
* Define & populate structured CSV per video
* Test GPT-4o prompts for tagging

**Phase 2:**

* RAG-based chatbot for teacher/leadership archives
* Quote and theme extraction from interviews

**Phase 3:**

* Unified interface for querying and tagging
* End-to-end AI-curated archive for storytelling and research
