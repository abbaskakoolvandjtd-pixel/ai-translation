import json
import json_repair  # pip install json-repair
import re
import os
import time
import difflib
import threading
import concurrent.futures
from openai import OpenAI
from typing import List, Dict, Any, Optional
from collections import Counter

from openai.types.shared_params import reasoning_effort
from shared_functions import count_tokens  # Your existing PyODBC connection function

from core.db import update_job_status

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_URL = "http://10.19.24.49:5090/v1"
API_KEY = "sk-N_j-qpRiMdEcN1bRhmnNiA"  # Must match config.yaml master_key

MAX_OUTPUT_TOKENS = 30000
INPUT_JSON = r"E:\Kakoolvand\pycharm_projects\markitdown_project\WDR 2026 Overview Booklet.json"
OUTPUT_JSON = r"E:\Kakoolvand\pycharm_projects\markitdown_project\unlimited_ocr_output_files\WDR 2026 Overview Booklet_persian.json"
MAX_CHUNK_TOKENS = 15000  # Dynamic chunking limit based on tokens

# Initialize OpenAI client for local server
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def _flatten_to_dicts(data: Any) -> list:
    """Recursively flattens nested lists to extract all dictionaries.
    Protects against LLM returning malformed nested JSON arrays (e.g., [[{...}]])"""
    result = []
    if isinstance(data, dict):
        result.append(data)
    elif isinstance(data, list):
        for item in data:
            result.extend(_flatten_to_dicts(item))
    return result


def parse_gemma_response(content):
    """Parse Gemma thinking and final answer"""
    if not content.startswith('<|channel>'):
        return {"thinking": None, "final_answer": content.strip(), "has_thinking": False}

    content_after_header = content[len('<|channel>'):].lstrip('\n')
    split_pattern = r'<channel\|>'
    match = re.search(split_pattern, content_after_header)

    if match:
        thinking = content_after_header.split(split_pattern)[0]
        answer = content_after_header.split(split_pattern)[-1]
        return {"thinking": thinking, "final_answer": answer, "has_thinking": True}

    return {"thinking": None, "final_answer": content_after_header.strip(), "has_thinking": False}


# ==============================================================================
# LLM INTERACTION & PARSING
# ==============================================================================
def call_llm(system_prompt: str, user_prompt: str, max_retries: int = 5) -> str:
    """Calls the local LLM with retry logic. Raises Exception if all retries fail."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    print("input tokens=",count_tokens(str(messages)))

    last_exception = None
    for attempt in range(max_retries + 1):
        MODEL = "qwen-3.8-27-ctx-temp"
        try:
            if attempt==max_retries:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    timeout=1000
                )
                raw = response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    extra_body={
                        "thinking_token_budget": 1024,
                        "chat_template_kwargs": {
                            "enable_thinking": True
                        },
                    },
                    timeout=1000
                )
                raw = parse_gemma_response(response.choices[0].message.content)["final_answer"]
            if not raw:
                raise ValueError("LLM returned empty final_answer")
            return raw

        except Exception as e:
            last_exception = e
            print(f"❌ LLM Error (Attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2)

    # 🔥 CRITICAL FIX: If we exit the loop, all retries failed.
    raise RuntimeError(f"LLM completely unavailable after {max_retries + 1} attempts. Last error: {last_exception}")


def parse_json_response(raw_content: str) -> Any:
    """Robust JSON parser that handles markdown code blocks, partial outputs, and malformed JSON."""
    if not raw_content:
        return None

    raw_content = raw_content.strip()

    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    elif raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]

    raw_content = raw_content.strip()

    # 1. Try standard JSON parsing
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        pass

    # 2. Fallback: Use json-repair to fix unescaped quotes (common with HTML tables)
    try:
        repaired = json_repair.loads(raw_content)
        if repaired:
            return repaired
    except Exception:
        pass

    # 3. Final regex fallback for partial outputs
    match = re.search(r'\[.*\]|\{.*\}', raw_content, re.DOTALL)
    if match:
        try:
            return json_repair.loads(match.group(0))
        except Exception:
            pass

    print(f"Warning: Failed to parse JSON from LLM response.")
    return None


# ==============================================================================
# TRANSLATION PIPELINE
# ==============================================================================
class TranslationPipeline:
    def __init__(self, input_path: str, output_path: str,
                 workers_merge_headers: int = 1,
                 workers_merge_paragraphs: int = 1,
                 workers_translate_headings: int = 1,
                 workers_translate_toc: int = 1,
                 workers_translate_body: int = 1,
                 job_id: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.data = None
        self.objects = []
        self.glossary = {}
        self.tm = {}
        self.headings_tm = {}
        self.section_map = {}

        # Parallelism settings
        self.workers_merge_headers = workers_merge_headers
        self.workers_merge_paragraphs = workers_merge_paragraphs
        self.workers_translate_headings = workers_translate_headings
        self.workers_translate_toc = workers_translate_toc
        self.workers_translate_body = workers_translate_body
        self.job_id = job_id

        # Thread lock for safe dictionary updates/reads
        self.lock = threading.Lock()

    def load_data(self):
        print("Loading JSON data...")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        obj_id = 0
        for page in self.data.get("pages", []):
            page_num = page.get("page")
            for obj in page.get("objects", []):
                if obj.get("type") in ["image", "page_number"] or not obj.get("content", "").strip() or not obj.get(
                        "bbox", None):
                    obj["obj_id"] = obj_id
                    obj["skip_translation"] = True
                else:
                    obj["obj_id"] = obj_id
                    obj["page_internal"] = page_num
                    obj["skip_translation"] = False
                    self.objects.append(obj)
                obj_id += 1
        print(f"Loaded {len(self.objects)} translatable objects.")

    def normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip().lower())

    def get_tm_matches(self, text: str, top_k: int = 2) -> List[Dict]:
        norm_text = self.normalize_text(text)

        with self.lock:
            if norm_text in self.tm:
                return [{"source": text, "translated": self.tm[norm_text]}]
            tm_snapshot = list(self.tm.items())

        matches = []
        for src, trans in tm_snapshot:
            ratio = difflib.SequenceMatcher(None, norm_text, src).ratio()
            if ratio > 0.75:
                matches.append((ratio, src, trans))

        matches.sort(reverse=True, key=lambda x: x[0])
        return [{"source": m[1], "translated": m[2]} for m in matches[:top_k]]

    def merge_split_headers_llm(self, max_workers=1):
        """Uses LLM with a 3-page sliding window to detect and merge split headers."""
        print(f"Merging split multi-line headers using LLM (workers: {max_workers})...")
        pages_dict = {}
        for obj in self.objects:
            p = obj.get("page_internal")
            if p not in pages_dict:
                pages_dict[p] = []
            pages_dict[p].append(obj)

        sorted_pages = sorted(pages_dict.keys())

        system_prompt = """# ROLE
You are an expert Document Layout Analyzer and OCR Post-Processing Specialist.

# CONTEXT
You will receive a JSON array of text blocks extracted from consecutive pages of a document. Due to OCR or layout extraction issues, single headings or titles are sometimes split across multiple consecutive text objects (e.g., "World" in one block, "Development" in the next).

# TASK
Identify consecutive short text blocks that actually form a single, split heading or title, and provide instructions to merge them.

# CONSTRAINTS & RULES
1. A split header typically consists of 2 to 4 consecutive short lines lacking terminal punctuation (periods, question marks).
2. Do NOT merge normal body paragraphs, bullet points, lists, or table rows.
3. The "merged_text" must combine the text with appropriate spacing (e.g., "World Development Report").

# OUTPUT FORMAT
Output STRICTLY a valid JSON array of merge instructions. Do not include markdown formatting, explanations, or conversational text.
Schema: [{"primary_id": <int>, "absorbed_ids": [<int>, ...], "merged_text": "<string>"}]
If no split headers are found, return an empty array: []"""

        tasks = []
        for i in range(len(sorted_pages)):
            context_pages = sorted_pages[i: i + 3]
            llm_input = []
            for p in context_pages:
                for obj in pages_dict[p]:
                    if not obj.get("skip_translation"):
                        if len(obj["content"].strip()) < 200:
                            llm_input.append({
                                "obj_id": obj["obj_id"],
                                "page": obj["page_internal"],
                                "type": obj.get("type"),
                                "content": obj["content"]
                            })
            tasks.append((i, llm_input))

        def process_task(task):
            idx, llm_input = task
            if not llm_input: return None
            user_prompt = f"<input_json>\n{json.dumps(llm_input, ensure_ascii=False)}\n</input_json>"
            raw = call_llm(system_prompt, user_prompt)
            return parse_json_response(raw)

        results = []
        if max_workers > 1 and tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_task, task) for task in tasks]
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    print(f"progress: {int(100 * (i + 1) / len(tasks))}%")
                    if self.job_id:
                        update_job_status(self.job_id, "TRANSLATING",
                                          status_detail=f"""در حال ترجمه فایل   /   ادغام عناوین ({int(100 * (i + 1) / len(tasks))}٪)""")
                    res = future.result()
                    if res: results.append(res)
                    if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                        print(f"  -> Header merge LLM calls progress: {int(100 * (i + 1) / len(tasks))}%")
        else:
            for i, task in enumerate(tasks):
                res = process_task(task)
                if res: results.append(res)

        # Apply merges sequentially to avoid object mutation race conditions
        processed_primary_ids = set()
        for parsed in results:
            if not parsed:
                continue

            for merge_instr in _flatten_to_dicts(parsed):
                p_id = merge_instr.get("primary_id")
                a_ids = merge_instr.get("absorbed_ids", [])
                m_text = merge_instr.get("merged_text", "")

                if p_id is None:
                    continue

                if p_id in processed_primary_ids:
                    continue

                primary_obj = next((o for o in self.objects if o["obj_id"] == p_id), None)
                if primary_obj:
                    processed_primary_ids.add(p_id)
                    primary_obj["content"] = m_text
                    primary_obj["merged_ids"] = [p_id] + a_ids

                    if primary_obj.get("type") not in ["title", "header"]:
                        primary_obj["type"] = "header"

                    for a_id in a_ids:
                        absorbed_obj = next((o for o in self.objects if o["obj_id"] == a_id), None)
                        if absorbed_obj:
                            absorbed_obj["skip_translation"] = True
                            absorbed_obj["merged_into"] = p_id

        print(f"Applied {len(processed_primary_ids)} header merges.")

    def merge_split_paragraphs_llm(self, max_workers=1):
        """Uses LLM to detect and merge body paragraphs that are split across objects or pages."""
        print(f"Merging split paragraphs across objects/pages using LLM (workers: {max_workers})...")

        system_prompt = """# ROLE
You are an expert Document Layout Analyzer and OCR Post-Processing Specialist.

# CONTEXT
You will receive a JSON array of text blocks. Due to layout extraction issues, a single continuous paragraph is sometimes split into multiple objects mid-sentence (e.g., a sentence breaks abruptly without punctuation, or a word is hyphenated across blocks).

# TASK
Identify consecutive text blocks that are actually fragments of a single continuous paragraph and provide instructions to merge them into a cohesive paragraph.

# CONSTRAINTS & RULES
1. ONLY merge if a sentence is clearly broken mid-sentence (e.g., lacks ending punctuation, ends with a lowercase letter, or continues a thought abruptly).
2. Do NOT merge distinct paragraphs, complete sentences, bullet points, lists, headings, or table rows.
3. Ensure the "merged_text" fixes the flow by adding proper spacing and removing inappropriate hyphens caused by the split.

# OUTPUT FORMAT
Output STRICTLY a valid JSON array of merge instructions. Do not include markdown formatting, explanations, or conversational text.
Schema: [{"primary_id": <int>, "absorbed_ids": [<int>, ...], "merged_text": "<string>"}]
If no split paragraphs are found, return an empty array: []"""

        body_objs = [o for o in self.objects if
                     not o.get("skip_translation") and o.get("type") not in ["title", "header"]]

        batch_size = 50
        step = 10

        tasks = []
        for i in range(0, len(body_objs), step):
            batch = body_objs[i: i + batch_size]
            if not batch: continue
            llm_input = [{"obj_id": o["obj_id"], "content": o["content"]} for o in batch]
            tasks.append(llm_input)

        def process_task(llm_input):
            user_prompt = f"<input_json>\n{json.dumps(llm_input, ensure_ascii=False)}\n</input_json>"
            raw = call_llm(system_prompt, user_prompt)
            return parse_json_response(raw)

        results = []
        if max_workers > 1 and tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_task, task) for task in tasks]
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    print(f"progress: {int(100 * (i + 1) / len(tasks))}%")
                    if self.job_id:
                        update_job_status(self.job_id, "TRANSLATING",
                                          status_detail=f"""در حال ترجمه فایل   /   ادغام پاراگراف ها ({int(100 * (i + 1) / len(tasks))}٪)""")
                    res = future.result()
                    if res: results.append(res)
                    if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                        print(f"  -> Paragraph merge LLM calls progress: {int(100 * (i + 1) / len(tasks))}%")
        else:
            for i, task in enumerate(tasks):
                res = process_task(task)
                if res: results.append(res)

        processed_primary_ids = set()
        for parsed in results:
            if not parsed:
                continue

            for merge_instr in _flatten_to_dicts(parsed):
                p_id = merge_instr.get("primary_id")
                a_ids = merge_instr.get("absorbed_ids", [])
                m_text = merge_instr.get("merged_text", "")

                if p_id is None:
                    continue

                primary_obj = next((o for o in self.objects if o["obj_id"] == p_id), None)
                if not primary_obj or p_id in processed_primary_ids or primary_obj.get(
                        "skip_translation") or primary_obj.get("merged_into"):
                    continue

                processed_primary_ids.add(p_id)
                primary_obj["content"] = m_text
                primary_obj["merged_ids"] = [p_id] + a_ids

                for a_id in a_ids:
                    absorbed_obj = next((o for o in self.objects if o["obj_id"] == a_id), None)
                    if absorbed_obj and not absorbed_obj.get("skip_translation") and not absorbed_obj.get(
                            "merged_into"):
                        absorbed_obj["skip_translation"] = True
                        absorbed_obj["merged_into"] = p_id

        print(f"Applied {len(processed_primary_ids)} paragraph merges.")

    def extract_glossary(self):
        print("Extracting initial glossary...")
        if self.job_id:
            update_job_status(self.job_id, "TRANSLATING",
                              status_detail=f"""در حال ترجمه فایل   /   در حال ایجاد واژه‌نامه برای لغات تخصصی...""")
        candidate_counter = Counter()
        for obj in self.objects:
            if obj.get("type") in ["title", "header", "text"] and len(obj["content"]) < 100:
                candidate_counter[obj["content"]] += 1

        most_common_candidates = [text for text, _ in candidate_counter.most_common(1000)]
        sample_text = "\n".join(most_common_candidates)

        system_prompt = """# ROLE
You are an expert Persian Terminologist and Translator specializing in technical, economic, and business documents.

# CONTEXT
You are building a master glossary for an English-to-Persian translation project. You will receive a list of candidate phrases from the document.

# TASK
Extract key technical terms, domain-specific jargon, acronyms, and proper nouns (organizations, specific programs). Provide their standard, professional Persian translations.

# CONSTRAINTS & RULES
1. Focus on high-value domain terms, not common English words.
2. For acronyms and organization names, append the English original in parentheses. Example: "هوش مصنوعی (AI)" or "بانک جهانی (World Bank)".
3. Ensure the Persian translation uses formal, academic, and standard terminology appropriate for official reports.
4. Extract at least 35 distinct terms if the text contains enough candidates.

# OUTPUT FORMAT
Output STRICTLY a valid JSON array of objects. Do not include markdown formatting, explanations, or conversational text.
Schema: [{"en": "<English term>", "fa": "<Persian translation>"}]"""

        user_prompt = f"<candidate_phrases>\n{sample_text}\n</candidate_phrases>"
        raw = call_llm(system_prompt, user_prompt)
        parsed = parse_json_response(raw)

        if parsed and isinstance(parsed, list):
            for item in _flatten_to_dicts(parsed):
                if "en" in item and "fa" in item:
                    self.glossary[item["en"].strip().lower()] = item["fa"]
        print(f"Initial Glossary extracted: {len(self.glossary)} terms.")

    def extract_chunk_glossary(self, chunk_texts: List[str]):
        """Extracts 5-10 new glossary terms from the current chunk, avoiding duplicates."""
        combined_text = "\n".join(chunk_texts)
        if len(combined_text) > 15000:
            combined_text = combined_text[:15000] + "..."

        system_prompt = """# ROLE
You are an expert Persian Terminologist and Translator.

# CONTEXT
You are dynamically expanding a master glossary for an ongoing translation task. You will receive the "Existing Glossary" and a new chunk of "Text to Analyze".

# TASK
Extract 5 to 10 NEW key technical terms, acronyms, or specific concepts from the "Text to Analyze" to add to the glossary.

# CONSTRAINTS & RULES
1. CRITICAL: You MUST NOT extract terms that are already present in the "Existing Glossary" (case-insensitive check).
2. For acronyms and organizations, append the English original in parentheses. Example: "توسعه پایدار (SD)".
3. Only extract highly relevant domain-specific terms, not common vocabulary.
4. If no new relevant terms are found in the text, return an empty array.

# OUTPUT FORMAT
Output STRICTLY a valid JSON array of objects. Do not include markdown formatting, explanations, or conversational text.
Schema: [{"en": "<English term>", "fa": "<Persian translation>"}]"""

        with self.lock:
            glossary_snapshot = dict(self.glossary)

        user_prompt = f"""<existing_glossary>
{json.dumps(glossary_snapshot, ensure_ascii=False)}
</existing_glossary>

<text_to_analyze>
{combined_text}
</text_to_analyze>"""

        raw = call_llm(system_prompt, user_prompt)
        parsed = parse_json_response(raw)

        if parsed and isinstance(parsed, list):
            with self.lock:
                new_terms_count = 0
                for item in _flatten_to_dicts(parsed):
                    if "en" in item and "fa" in item:
                        en_term = item["en"].strip().lower()
                        if en_term and en_term not in self.glossary:
                            self.glossary[en_term] = item["fa"]
                            new_terms_count += 1
                if new_terms_count > 0:
                    print(f"  -> Added {new_terms_count} new terms to glossary from current chunk.")

    def chunk_and_translate(self, objs, system_prompt, context_type="body", max_workers=1):
        """Handles exact match deduplication and dynamic token-based chunking."""
        untranslated_objs = []
        for obj in objs:
            norm_text = self.normalize_text(obj["content"])
            if norm_text in self.tm:
                obj["translated_content"] = self.tm[norm_text]
            else:
                untranslated_objs.append(obj)

        if not untranslated_objs:
            return

        overhead_text = system_prompt + json.dumps(self.glossary, ensure_ascii=False)
        PROMPT_OVERHEAD = count_tokens(overhead_text) + 1000

        chunks = []
        current_chunk = []
        current_tokens = 0

        for ind, obj in enumerate(untranslated_objs):
            obj_tokens = count_tokens(obj["content"]) + 30

            if current_tokens + obj_tokens + PROMPT_OVERHEAD > MAX_CHUNK_TOKENS and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append(obj)
            current_tokens += obj_tokens

        if current_chunk:
            chunks.append(current_chunk)

        if not chunks: return

        def process_chunk(chunk):
            self._process_llm_chunk(chunk, system_prompt, context_type)

        print(f"Translating {len(chunks)} chunks of {context_type} with {max_workers} workers...")

        if max_workers > 1 and len(chunks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    print(f"progress: {int(100 * (i + 1) / len(chunks))}%")
                    if self.job_id:
                        if context_type == "body":
                            text_status = "متن اصلی"
                        elif context_type == "toc":
                            text_status = "فهرست مطالب"
                        elif context_type == "headings":
                            text_status = "عناوین"

                        update_job_status(self.job_id, "TRANSLATING",
                                          status_detail=f"""در حال ترجمه فایل   /   ترجمه {text_status} ({int(100 * (i + 1) / len(chunks))}٪)""")
                    future.result()
                    print(f"  -> {context_type.capitalize()} translation progress: {int(100 * (i + 1) / len(chunks))}%")
        else:
            for i, chunk in enumerate(chunks):
                process_chunk(chunk)
                print(f"  -> {context_type.capitalize()} translation progress: {int(100 * (i + 1) / len(chunks))}%")

    def _process_llm_chunk(self, chunk, system_prompt, context_type):
        if context_type == "body":
            chunk_texts = [obj["content"] for obj in chunk]
            self.extract_chunk_glossary(chunk_texts)

        with self.lock:
            glossary_str = json.dumps(self.glossary, ensure_ascii=False)

        if context_type == "body":
            tm_context = []
            for obj in chunk:
                tm_context.extend(self.get_tm_matches(obj["content"]))
            unique_tm = {m['source']: m for m in tm_context}.values()

            with self.lock:
                tm_str = json.dumps(list(unique_tm), ensure_ascii=False)
            section_context = self.section_map.get(chunk[0]["obj_id"], "Unknown")

            segments = "\n".join([f"ID: {obj['obj_id']} | Content: {obj['content']}" for obj in chunk])
            user_prompt = f"""<context>Current Section: {section_context}</context>
<glossary>{glossary_str}</glossary>
<translation_memory>{tm_str}</translation_memory>
<segments_to_translate>
{segments}
</segments_to_translate>"""

        elif context_type == "toc":
            tm_context = []
            for obj in chunk:
                clean_content = re.sub(r'[\s\d\.]+$', '', obj["content"]).strip()
                norm_clean = self.normalize_text(clean_content)
                with self.lock:
                    if norm_clean in self.headings_tm:
                        tm_context.append({"source": obj["content"], "translated": self.headings_tm[norm_clean]})
            tm_str = json.dumps(tm_context, ensure_ascii=False)

            entries = "\n".join([f"ID: {obj['obj_id']} | Content: {obj['content']}" for obj in chunk])
            user_prompt = f"""<glossary>{glossary_str}</glossary>
<translation_memory>{tm_str}</translation_memory>
<toc_entries>
{entries}
</toc_entries>"""

        else:  # headings
            headings = "\n".join([f"ID: {obj['obj_id']} | Content: {obj['content']}" for obj in chunk])
            user_prompt = f"""<glossary>{glossary_str}</glossary>
<headings_to_translate>
{headings}
</headings_to_translate>"""

        raw = call_llm(system_prompt, user_prompt)
        parsed = parse_json_response(raw)

        translated_ids = set()
        if parsed and isinstance(parsed, list):
            with self.lock:
                for res in _flatten_to_dicts(parsed):
                    obj_id = res.get("id")
                    trans = res.get("translated")
                    if obj_id is not None and trans:
                        translated_ids.add(obj_id)
                        orig = next((o for o in self.objects if o["obj_id"] == obj_id), None)
                        if orig:
                            orig["translated_content"] = trans
                            self.tm[self.normalize_text(orig["content"])] = trans

                            if orig.get("type") in ["title", "header"]:
                                self.headings_tm[self.normalize_text(orig["content"])] = trans

        # --- RETRY LOGIC FOR SKIPPED IDS ---
        missing_objs = [obj for obj in chunk if obj["obj_id"] not in translated_ids]
        if missing_objs:
            print(f"  -> Warning: LLM skipped {len(missing_objs)} IDs (likely captions/tables). Retrying...")
            missing_segments = "\n".join([f"ID: {obj['obj_id']} | Content: {obj['content']}" for obj in missing_objs])
            retry_prompt = f"""CRITICAL: You missed translating the following segments in your previous response. 
Translate them NOW into Persian and output ONLY the valid JSON array. Ensure HTML tags and quotes are perfectly preserved and escaped.
<segments_to_translate>
{missing_segments}
</segments_to_translate>"""

            retry_raw = call_llm(system_prompt, retry_prompt)
            retry_parsed = parse_json_response(retry_raw)

            if retry_parsed and isinstance(retry_parsed, list):
                with self.lock:
                    for res in _flatten_to_dicts(retry_parsed):
                        obj_id = res.get("id")
                        trans = res.get("translated")
                        if obj_id is not None and trans:
                            orig = next((o for o in self.objects if o["obj_id"] == obj_id), None)
                            if orig:
                                orig["translated_content"] = trans
                                self.tm[self.normalize_text(orig["content"])] = trans
                                if orig.get("type") in ["title", "header"]:
                                    self.headings_tm[self.normalize_text(orig["content"])] = trans

    def translate_headings(self, max_workers=1):
        print("Translating headings...")
        system_prompt = """# ROLE
You are an expert English-to-Persian Translator specializing in formal business and technical documents.

# TASK
Translate the provided document headings and titles into fluent, natural, and professional Persian.

# CONSTRAINTS & RULES
1. Keep translations concise, impactful, and title-cased (appropriate for headings).
2. STRICTLY use the provided Glossary terms where applicable.
3. For technical terms or acronyms not in the glossary, use the Persian translation followed by the English acronym in parentheses. Example: "شاخص توسعه انسانی (HDI)".
4. Maintain the original meaning without adding or omitting words.

# OUTPUT FORMAT
Output STRICTLY a valid JSON array of objects. Do not include markdown formatting, explanations, or conversational text.
Schema: [{"id": <int>, "translated": "<Persian text>"}]"""

        headings = [obj for obj in self.objects if
                    obj.get("type") in ["title", "header"] and not obj.get("skip_translation")]

        self.chunk_and_translate(headings, system_prompt, context_type="headings", max_workers=max_workers)

        current_section = "مقدمه"
        for obj in self.objects:
            if obj.get("type") == "title" and "translated_content" in obj:
                current_section = obj["translated_content"]
            self.section_map[obj["obj_id"]] = current_section

    def translate_toc(self, max_workers=1):
        print("Translating Table of Contents...")
        if self.job_id:
            update_job_status(self.job_id, "TRANSLATING",
                              status_detail=f"""در حال ترجمه فایل   /   ترجمه فهرست مطالب... """)
        toc_objs = []
        for obj in self.objects:
            if obj.get("page_internal") in [1, 2] and obj.get("type") in ["text", "title"] and not obj.get(
                    "skip_translation"):
                content = obj["content"]
                clean_content = re.sub(r'[\s\d\.]+$', '', content).strip()
                norm_clean = self.normalize_text(clean_content)
                if norm_clean in self.headings_tm or re.search(r'\d+$', content.strip()):
                    toc_objs.append(obj)

        system_prompt = """# ROLE
You are an expert English-to-Persian Translator specializing in formal document formatting.

# TASK
Translate the provided Table of Contents (TOC) entries into Persian while preserving the structural integrity of the TOC.

# CONSTRAINTS & RULES
1. CRITICAL: You MUST use the EXACT Persian translation for section titles that already exist in the provided Translation Memory. Consistency is paramount.
2. DO NOT translate or alter the page numbers at the end of the string. Keep them exactly as they appear.
3. Maintain any formatting dots, dashes, or spacing between the title and the page number.
4. Use the provided Glossary for any new terms.

# OUTPUT FORMAT
Output STRICTLY a valid JSON array of objects. Do not include markdown formatting, explanations, or conversational text.
Schema: [{"id": <int>, "translated": "<Persian text>"}]"""

        self.chunk_and_translate(toc_objs, system_prompt, context_type="toc", max_workers=max_workers)

    def translate_body(self, max_workers=1):
        print("Translating body paragraphs...")
        toc_ids = set(obj["obj_id"] for obj in self.objects if "translated_content" in obj)
        body_objs = [obj for obj in self.objects if
                     (obj.get("type") == "text" or obj.get("type") == "image_footnote" or obj.get(
                         "type") == "image_caption" or obj.get("type") == "table") and not obj.get(
                         "skip_translation") and obj["obj_id"] not in toc_ids]

        system_prompt = """# ROLE
You are an expert English-to-Persian Translator specializing in formal, academic, and technical documents (e.g., World Development Reports).

# CONTEXT
You will receive the current document Section Context, a Glossary, Translation Memory (similar previously translated phrases), and a list of text segments to translate.

# TASK
Translate the provided text segments into highly fluent, natural, and academically rigorous Persian.

# CONSTRAINTS & RULES
1. ACCURACY: Maintain the exact meaning. Do not hallucinate, add, or omit information.
2. GLOSSARY: STRICTLY use the provided Glossary terms. Do not invent alternative translations for glossary terms.
3. CONSISTENCY: Use the Translation Memory to ensure phrasing consistency with previously translated segments.
4. TERMINOLOGY: For new technical terms not in the glossary, use the Persian translation followed by the English term in parentheses on first occurrence.
5. INVARIANTS: Keep all numbers, dates, URLs, and statistical formats exactly unchanged.
6. TONE: Use formal, objective, and professional Persian suitable for an official international report.
7. NO SKIPPING: You MUST translate EVERY single ID provided, including "Figure X" captions, "Source:" footnotes, and HTML tables. Do not omit any IDs.
8. HTML TABLES: If translating HTML, preserve all tags exactly as they are and ONLY translate the visible text inside the tags. Ensure all quotes inside the HTML are properly escaped for JSON.

# OUTPUT FORMAT
Output STRICTLY a valid JSON array of objects. Do not include markdown formatting, explanations, or conversational text.
Schema: [{"id": <int>, "translated": "<Persian text>"}]"""

        self.chunk_and_translate(body_objs, system_prompt, context_type="body", max_workers=max_workers)

    def reconstruct_and_save(self):
        print("Reconstructing JSON...")
        if self.job_id:
            update_job_status(self.job_id, "TRANSLATING",
                              status_detail=f"""در حال ترجمه فایل   /   ذخیره‌سازی نتایج ترجمه... """)
        for page in self.data.get("pages", []):
            new_objects = []
            for obj in page.get("objects", []):
                if obj.get("merged_into") is not None:
                    continue

                if "translated_content" in obj:
                    obj["content_fa"] = obj["translated_content"]
                    del obj["translated_content"]

                for key in ["obj_id", "skip_translation", "page_internal", "merged_ids", "merged_into"]:
                    if key in obj:
                        del obj[key]
                new_objects.append(obj)
            page["objects"] = new_objects

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved translated JSON to {self.output_path}")

    def _format_elapsed(self, seconds_int: int) -> str:
        """Convert a duration in seconds to a human‑readable Hh Mm Ss.s string."""
        hrs = int(seconds_int // 3600)
        mins = int((seconds_int % 3600) // 60)
        secs = seconds_int % 60
        parts = []
        if hrs:
            parts.append(f"{hrs}h")
        if mins or hrs:
            parts.append(f"{mins}m")
        parts.append(f"{secs:.2f}s")
        return " ".join(parts)

    def run(self):
        total_time = 0
        tic = time.time()
        self.load_data()
        toc = time.time()
        load_elapsed = int(toc - tic)
        total_time += load_elapsed
        print(f"load_data time: {self._format_elapsed(load_elapsed)} / total seconds: {load_elapsed}")

        tic = time.time()
        self.merge_split_headers_llm(max_workers=self.workers_merge_headers)
        toc = time.time()
        load_elapsed = int(toc - tic)
        total_time += load_elapsed
        print(f"merge_split_headers_llm time: {self._format_elapsed(load_elapsed)} / total seconds: {load_elapsed}")

        tic = time.time()
        self.merge_split_paragraphs_llm(max_workers=self.workers_merge_paragraphs)
        toc = time.time()
        load_elapsed = int(toc - tic)
        total_time += load_elapsed
        print(f"merge_split_paragraphs_llm time: {self._format_elapsed(load_elapsed)} / total seconds: {load_elapsed}")

        tic = time.time()
        self.extract_glossary()
        toc = time.time()
        load_elapsed = int(toc - tic)
        total_time += load_elapsed
        print(f"extract_glossary time: {self._format_elapsed(load_elapsed)} / total seconds: {load_elapsed}")

        tic = time.time()
        self.translate_headings(max_workers=self.workers_translate_headings)
        toc = time.time()
        load_elapsed = int(toc - tic)
        total_time += load_elapsed
        print(f"translate_headings time: {self._format_elapsed(load_elapsed)} / total seconds: {load_elapsed}")

        tic = time.time()
        self.translate_toc(max_workers=self.workers_translate_toc)
        toc = time.time()
        load_elapsed = int(toc - tic)
        total_time += load_elapsed
        print(f"translate_toc time: {self._format_elapsed(load_elapsed)} / total seconds: {load_elapsed}")

        tic = time.time()
        self.translate_body(max_workers=self.workers_translate_body)
        toc = time.time()
        load_elapsed = int(toc - tic)
        total_time += load_elapsed
        print(f"translate_body time: {self._format_elapsed(load_elapsed)} / total seconds: {load_elapsed}")

        tic = time.time()
        self.reconstruct_and_save()
        toc = time.time()
        load_elapsed = int(toc - tic)
        total_time += load_elapsed
        print(f"reconstruct_and_save time: {self._format_elapsed(load_elapsed)} / total seconds: {load_elapsed}")

        print(f"Total process time: {self._format_elapsed(total_time)} / total seconds: {total_time}")


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found. Please place your OCR output JSON in the same directory.")
    else:
        pipeline = TranslationPipeline(
            INPUT_JSON,
            OUTPUT_JSON,
            workers_merge_headers=8,
            workers_merge_paragraphs=4,
            workers_translate_headings=4,
            workers_translate_toc=4,
            workers_translate_body=3
        )
        pipeline.run()