# Methodological Note: LLM Role Allocation in Multi-Agent Clinical Evaluation
**Project:** Conditional Sycophancy & Demographic Bias in Clinical AI
**Research Context:** Mitacs Globalink Research Internship, Brock University 
**Prepared for:** Dr. Naser Ezzati-Jivan
**Date:** September 2026

## 1. Objective
To document the architectural decision regarding model selection for the orchestrator agents (Generator and Validator) versus the test subject agent (Agent B) in the Phase 0 clinical bias pipeline.

## 2. Empirical Observation: The Instruction-Following Gap
During initial pipeline validation for the ESI triage and USMLE diagnostic tasks, Gemini-3.8-Flash and DeepSeek-V4-Flash were tested in the Generator (Agent A) and Validator roles. 

These specific models exhibited an inability to adhere strictly to formatting and preservation constraints during the generation of adversarial clinical vignettes. Specifically, they repeatedly triggered infinite validation loops by:
*   Failing to accurately preserve baseline vital signs and clinical facts (failing `CHECK 3`).
*   Dropping or heavily mutating the injected demographic carrier lines (failing `CHECK 2`).
*   Appending conversational filler that disrupted downstream token-edit-distance (paraphrase) calculations.

Conversely, OpenAI's GPT-5.6-Luna executed these roles flawlessly, passing the rigid validation checks on the first attempt and ensuring perfect clinical consistency across the $E+$ and $E-$ variants.

## 3. Alignment with Existing Literature
While our empirical observation is currently limited to Gemini-3.8-Flash and DeepSeek-V4-Flash, this behavioral divergence aligns with broader trends identified in recent medical AI benchmarking. The authors of the FairMedQA benchmark evaluated different candidate models for their multi-agent adversarial generation framework and indicated that GPT models demonstrated superior instruction-following capabilities in maintaining clinical consistency compared to models like DeepSeek. Consequently, GPT was adopted as the default foundation model for their generation framework[cite: 1].

## 4. Pipeline Architecture Decision
Based on our specific local observations and supporting literature, the pipeline enforces a strict bifurcation of model roles to ensure high-fidelity evaluation:
*   **Orchestration (Agent A & Validator):** Restricted exclusively to GPT-5.6-Luna (via OpenRouter) to guarantee that the adversarial dataset is clinically sound, perfectly constrained, and free of prompt-induced artifacts.
*   **Evaluation (Agent B):** Gemini-3.8-Flash and DeepSeek-V4-Flash are strictly confined to the role of the test subject. 

By isolating the generative tasks to the most reliable instruction-follower, we ensure that any measured demographic bias or conditional sycophancy in the test subjects is a true reflection of their internal mechanistic flaws, rather than a byproduct of a poorly constructed prompt.