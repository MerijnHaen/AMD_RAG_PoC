import openai
from datetime import datetime
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import faiss
import time
import json
import os
!pip install faiss-cpu sentence-transformers - -quiet


# !!IMPORTANT TO ABSTRACT THIS AWAY IN PRODUCTION CODE!!
openai.api_key = ""


class SmartRAGExpander:
    def __init__(self, embedder_model='all-MiniLM-L6-v2', faiss_dim=384, backup_folder="data/"):
        self.embedder = SentenceTransformer(embedder_model)
        self.index = faiss.IndexFlatL2(faiss_dim)
        self.metadata = []
        self.existing_pmids = set()
        self.rejected_pmids = set()
        self.lineage = {}
        self.backup_folder = backup_folder

        os.makedirs(self.backup_folder, exist_ok=True)
        self.load_latest_snapshot()
        self.load_rejected_pmids()

    # RECURSIVELY GET NEW PAPERS AND ABSTRACTS
    def fetch_related_pmids(self, pmid):
        import requests
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        params = {
            "dbfrom": "pubmed",
            "linkname": "pubmed_pubmed",
            "id": pmid,
            "retmode": "json"
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            links = []
            for linkset in data.get("linksets", []):
                for link in linkset.get("linksetdbs", []):
                    if link.get("linkname") == "pubmed_pubmed":
                        links.extend(link.get("links", []))
            return list(set(links))
        except Exception as e:
            print(f"Error fetching related PMIDs for {pmid}: {e}")
            return []

    def fetch_pubmed_abstract(self, pmid):
        import requests
        import xml.etree.ElementTree as ET
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            article = root.find(".//Article")
            if article is None:
                return None
            title = article.findtext(".//ArticleTitle", default="")
            abstract_elements = article.findall(".//AbstractText")
            if not abstract_elements:
                return None
            abstract = " ".join([el.text or "" for el in abstract_elements])
            return {"pmid": pmid, "title": title.strip(), "abstract": abstract.strip()}
        except Exception as e:
            print(f"Error fetching abstract for PMID {pmid}: {e}")
            return None

    def selective_expand_recursive(self, seed_pmids, max_depth=2, max_total=1000, delay=0.45, filter_keywords=None):
        if filter_keywords is None:
            filter_keywords = [
                "macular", "retina", "retinal", "age-related macular degeneration",
                "geographic atrophy", "anti-VEGF", "dry AMD", "wet AMD", "choroidal neovascularization"
            ]

        visited = set(seed_pmids)
        to_visit = list(seed_pmids)
        depth = 0
        total_added = len(self.metadata)

        while to_visit and total_added < max_total and depth < max_depth:
            print(f"🔍 Depth {depth} | Expanding {len(to_visit)} seeds...")
            next_visit = []

            for seed in to_visit:
                if total_added >= max_total:
                    break

                related = self.fetch_related_pmids(seed)

                # ✅ NEW: skip both already-added and rejected PMIDs
                new_pmids = [
                    pmid for pmid in related
                    if pmid not in self.existing_pmids and pmid not in self.rejected_pmids
                ]

                for pmid in new_pmids:
                    if total_added >= max_total:
                        break

                    paper = self.fetch_pubmed_abstract(pmid)
                    if not paper:
                        continue

                    title = paper["title"].lower()
                    abstract = paper["abstract"].lower()

                    if not any(kw in abstract or kw in title for kw in filter_keywords):
                        print(
                            f"🛑 Skipping (irrelevant) {pmid}: '{title[:60]}...'")
                        # ✅ NEW: Remember this PMID as rejected forever
                        self.rejected_pmids.add(pmid)
                        continue

                    full_text = paper["title"] + " " + paper["abstract"]
                    embedding = self.embedder.encode(full_text)
                    self.index.add(np.array([embedding], dtype=np.float32))
                    self.metadata.append(paper)
                    self.existing_pmids.add(pmid)
                    self.lineage.setdefault(pmid, []).append(seed)
                    total_added += 1
                    print(
                        f"✅ {total_added}. Added {pmid}: {paper['title'][:60]}...")
                    time.sleep(delay)

                    next_visit.append(pmid)

            to_visit = next_visit
            depth += 1

        print(
            f"✅ Selective expansion complete. Total documents now: {len(self.metadata)}")

    def expand_from_current_db(self, max_depth=2, max_total=1000, delay=0.45, filter_keywords=None):
        seed_pmids = [paper["pmid"] for paper in self.metadata]
        self.selective_expand_recursive(
            seed_pmids, max_depth=max_depth, max_total=max_total, delay=delay, filter_keywords=filter_keywords)

    # FILTERING
    def filter_and_tag_metadata(self, min_tokens=100, keywords=None, drop_irrelevant=True, verbose=True):
        if keywords is None:
            keywords = [
                "macular", "retina", "retinal", "age-related macular degeneration",
                "geographic atrophy", "anti-VEGF", "dry AMD", "wet AMD", "choroidal neovascularization"
            ]

        filtered = []
        dropped = []
        seen_pmids = set()
        total_checked = 0

        for paper in self.metadata:
            total_checked += 1
            pmid = paper.get("pmid")
            if pmid in seen_pmids:
                if verbose:
                    print(f"🗑 Duplicate PMID {pmid} skipped")
                continue
            seen_pmids.add(pmid)

            abstract = paper.get("abstract", "").strip()
            token_count = len(re.findall(r'\w+', abstract))

            if token_count < min_tokens:
                if verbose:
                    print(
                        f"🗑 Dropping (too short) PMID {pmid}: {token_count} tokens")
                dropped.append(paper)
                continue

            tags = [kw for kw in keywords if kw.lower() in abstract.lower()]
            paper["tags"] = tags

            if drop_irrelevant and not tags:
                if verbose:
                    print(
                        f"🗑 Dropping (no keyword match) PMID {pmid}: '{paper['title'][:50]}'")
                dropped.append(paper)
                self.rejected_pmids.add(pmid)
                continue

            if verbose:
                print(f"✅ Keeping PMID {pmid} | Tags: {tags}")
            filtered.append(paper)

        self.metadata = filtered
        print(
            f"\n🔎 Filtered {len(filtered)} papers out of {total_checked}. Dropped {len(dropped)}. Duplicates silently removed.")

    def semantic_filter_with_gpt(self, openai_api_key=openai.api_key, model="gpt-3.5-turbo", verbose=True, overwrite=False):

        relevant_papers = []
        skipped = 0
        total_checked = 0

        for paper in self.metadata:
            total_checked += 1
            # Skip already tagged papers unless overwrite=True
            if not overwrite and "semantic_gpt_relevance" in paper:
                if paper["semantic_gpt_relevance"] == "YES":
                    relevant_papers.append(paper)
                    if verbose:
                        print(
                            f"✅ Already marked relevant: {paper['title'][:60]}")
                else:
                    if verbose:
                        print(
                            f"🗑 Already marked irrelevant: {paper['title'][:60]}")
                continue

            abstract = paper.get("abstract", "")
            prompt = f"""
You are an ophthalmology expert. 
Is the following study relevant to age-related macular degeneration (AMD), geographic atrophy (GA), or retinal degeneration? 
Reply only with 'YES' or 'NO'.

Abstract:
{abstract}
"""

            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0
                )
                answer = response['choices'][0]['message']['content'].strip(
                ).upper()
                if answer.startswith("YES"):
                    relevant_papers.append(paper)
                    if verbose:
                        print(f"✅ Kept: {paper['title'][:60]}...")
                else:
                    self.rejected_pmids.add(pmid)
                    if verbose:
                        print(f"🗑 Discarded: {paper['title'][:60]}...")
            except Exception as e:
                print(f"⚠️ Error filtering paper {paper['pmid']}: {e}")
                skipped += 1

        self.metadata = relevant_papers
        print(
            f"\n🎯 Semantic filtering complete. Checked {total_checked} papers. Kept {len(relevant_papers)}. Skipped {skipped} due to errors.")

    # Semantic Categorization & Tagging
    def enrich_papers_semantically(self, openai_api_key, model="gpt-3.5-turbo", overwrite=False, max_docs=None, verbose=True):
        count = 0
        updated = 0

        for paper in self.metadata:
            if max_docs is not None and count >= max_docs:
                break

            if all(k in paper for k in ["paper_type", "drug_names", "trial_phase"]) and not overwrite:
                if verbose:
                    print(f"✅ Already enriched: {paper['title'][:60]}")
                count += 1
                continue

            abstract = paper.get("abstract", "")
            prompt = f"""
    You are a biomedical domain expert. Please extract structured metadata from the following abstract.

    Return your response in the following JSON format:
    {{
      "paper_type": "[Clinical Trial / Mechanism Study / Review or Commentary / Other]",
      "drug_names": ["<list of drug names mentioned>"],
      "trial_phase": "[Phase 1 / Phase 2 / Phase 3 / Not applicable / Unknown]"
    }}

    If there are no drug names, return an empty list. If the phase is unknown or not relevant, use "Not applicable" or "Unknown".

    Abstract:
    {abstract}
    """

            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0
                )
                content = response['choices'][0]['message']['content'].strip()

                # Attempt to parse it as JSON directly
                try:
                    parsed = json.loads(content)
                    paper["paper_type"] = parsed.get("paper_type", "Unknown")
                    paper["drug_names"] = parsed.get("drug_names", [])
                    paper["trial_phase"] = parsed.get("trial_phase", "Unknown")
                except Exception as e:
                    # fallback if GPT returns malformed JSON
                    paper["paper_type"] = "Unknown"
                    paper["drug_names"] = []
                    paper["trial_phase"] = "Unknown"
                    print(f"⚠️ GPT returned unparseable result: {content}")

                updated += 1
                if verbose:
                    print(
                        f"📚 Enriched: {paper['title'][:60]} → Type: {paper['paper_type']} | Drugs: {paper['drug_names']} | Phase: {paper['trial_phase']}")
            except Exception as e:
                print(f"⚠️ GPT error for PMID {paper.get('pmid')}: {e}")

            count += 1

        print(f"\n✅ Semantic enrichment complete. {updated} papers enriched.")

    # GPT-aware contextual answer generator,
    # dynamically assembling the context by paper type and annotating trials with drug names and phases.
    def answer_with_gpt_contextual(self, query, retrieved_papers, openai_api_key, model="gpt-3.5-turbo", verbose=True):

        # Group by paper_type
        trials = [p for p in retrieved_papers if p.get(
            "paper_type") == "Clinical Trial"]
        mechanisms = [p for p in retrieved_papers if p.get(
            "paper_type") == "Mechanism Study"]
        reviews = [p for p in retrieved_papers if p.get(
            "paper_type") == "Review or Commentary"]
        other = [p for p in retrieved_papers if p.get("paper_type") not in [
            "Clinical Trial", "Mechanism Study", "Review or Commentary"]]

        context_blocks = []

        if trials:
            context_blocks.append(f"🔬 Clinical Trials ({len(trials)} papers):")
            for paper in trials:
                drug_str = ", ".join(
                    paper.get("drug_names", [])) or "Unknown drugs"
                trial_phase = paper.get("trial_phase", "Unknown phase")
                context_blocks.append(
                    f"- {paper['title']} [Drugs: {drug_str} | Phase: {trial_phase}]\nAbstract: {paper['abstract']}"
                )

        if mechanisms:
            context_blocks.append(
                f"\n🧠 Mechanism Studies ({len(mechanisms)} papers):")
            for paper in mechanisms:
                context_blocks.append(
                    f"- {paper['title']}\nAbstract: {paper['abstract']}")

        if reviews:
            context_blocks.append(
                f"\n📚 Reviews and Commentary ({len(reviews)} papers):")
            for paper in reviews:
                context_blocks.append(
                    f"- {paper['title']}\nAbstract: {paper['abstract']}")

        if other:
            context_blocks.append(f"\n📁 Other Papers ({len(other)} papers):")
            for paper in other:
                context_blocks.append(
                    f"- {paper['title']}\nAbstract: {paper['abstract']}")

        context_text = "\n\n".join(context_blocks)

        prompt = f"""
    You are a biomedical research assistant for pharmaceutical executives and medical directors.

    Query: {query}

    Your job is to provide a concise, structured summary based on the following studies. Clearly distinguish between clinical trials, mechanism studies, and reviews. Highlight drug names, trial phases, and notable results where possible. Use plain language suited for a strategic audience.

    Context:
    {context_text}

    Answer:
    """

        if verbose:
            print("📤 Prompt assembled. Sending to GPT...")

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            answer = response["choices"][0]["message"]["content"]
            print("✅ GPT response received.")
            return answer

        except Exception as e:
            print(f"⚠️ GPT error: {e}")
            return None

    # Abstract away the prompting magic and just open up the chatbox using "ask_question()"
    def ask_question(self, query, top_k=10, openai_api_key=None, model="gpt-3.5-turbo", verbose=True, return_sources=True):
        """
        Full-stack query handler:
        - Embeds query
        - Retrieves top_k papers from FAISS
        - Generates contextual prompt
        - Sends to GPT
        - Returns answer (+ optionally source documents)
        """
        if openai_api_key is None:
            raise ValueError("Please provide an OpenAI API key.")

        if verbose:
            print(f"🤖 Asking: '{query}'")
            print(f"🔎 Retrieving top {top_k} relevant papers from FAISS...")

        retrieved_papers = self.query_vector_db(query, k=top_k)

        if not retrieved_papers:
            return ("⚠️ No relevant documents found for this query.", [])

        if verbose:
            print(f"📚 Passing retrieved papers to GPT for contextual answer...")

        answer = self.answer_with_gpt_contextual(
            query=query,
            retrieved_papers=retrieved_papers,
            openai_api_key=openai_api_key,
            model=model,
            verbose=verbose
        )

        if return_sources:
            return answer, retrieved_papers
        else:
            return answer

    # Take care of the database retrieval
    def query_vector_db(self, question, k=5):
        query_embedding = self.embedder.encode(question)
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), k)
        results = [self.metadata[i]
                   for i in indices[0] if i < len(self.metadata)]
        return results

    # DISK ACTIONS
    def save_rejected_pmids(self):
        path = os.path.join(self.backup_folder, "rejected_pmids.json")
        with open(path, "w") as f:
            json.dump(list(self.rejected_pmids), f)

    def load_rejected_pmids(self):
        path = os.path.join(self.backup_folder, "rejected_pmids.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.rejected_pmids = set(json.load(f))
            print(f"📥 Loaded {len(self.rejected_pmids)} rejected PMIDs.")
        else:
            print("ℹ️ No rejected PMIDs file found. Starting clean.")

    def save_current_snapshot(self, keep_latest=3, backup_folder="data/"):
        print("🔄 Rebuilding FAISS index before saving...")
        self.index = faiss.IndexFlatL2(384)
        for paper in self.metadata:
            full_text = paper["title"] + " " + paper["abstract"]
            embedding = self.embedder.encode(full_text)
            self.index.add(np.array([embedding], dtype=np.float32))

        os.makedirs(backup_folder, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M")
        index_file = f"{backup_folder}smart_index_{now}.faiss"
        metadata_file = f"{backup_folder}smart_metadata_{now}.json"

        faiss.write_index(self.index, index_file)
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f)
        self.save_rejected_pmids()
        print(f"✅ Snapshot saved: {index_file} | {metadata_file}")

    def load_latest_snapshot(self):
        files = os.listdir(self.backup_folder)
        index_files = sorted([f for f in files if f.startswith(
            "smart_index") and f.endswith(".faiss")])
        metadata_files = sorted([f for f in files if f.startswith(
            "smart_metadata") and f.endswith(".json")])
        if index_files and metadata_files:
            latest_index = os.path.join(self.backup_folder, index_files[-1])
            latest_meta = os.path.join(self.backup_folder, metadata_files[-1])
            self.index = faiss.read_index(latest_index)
            with open(latest_meta, "r") as f:
                self.metadata = json.load(f)
            self.existing_pmids = set(p["pmid"] for p in self.metadata)
            print(f"📦 Loaded latest snapshot: {latest_index}, {latest_meta}")
        else:
            print("ℹ️ No previous snapshot found. Starting fresh.")


# Instantiate the final class
smart_expander = SmartRAGExpander()
