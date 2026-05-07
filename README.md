# CAPR Search

Plain-language Q&A over Civil Air Patrol regulations and pamphlets. Live at <https://capr.freshskyai.com>.

CAP members ask a question; the LLM answers using a curated index of active CAPRs/CAPPs + general training-data knowledge, citing the relevant regulation. Verify-current-version reminder always appears.

Standalone Flask app, no `freshsky_common` dependency. LLM auto-fallback chain (Groq → Cerebras → Gemini → Mistral → HuggingFace). No PII, no member data, no eServices integration.
