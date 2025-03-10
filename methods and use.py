1. instantiate smart_expander

2. forage new studies on the basis of what's already in the database and certain keywords
seed_pmids = [paper["pmid"] for paper in smart_expander.metadata]  # Expand from existing base
smart_expander.selective_expand_recursive(seed_pmids, max_depth=1, max_total=1250)

3. filter 1
smart_expander.filter_and_tag_metadata()

4. filter 2, use chatgpt to decide whether the study is relevant at all
smart_expander.semantic_filter_with_gpt()

5. check whether there are papers in the database
print(f"Total papers in metadata: {len(smart_expander.metadata)}")

6. save current database snapshot
smart_expander.save_current_snapshot()

7. enrich the information in the papers semantically, for easier searching and grouping later on
smart_expander.enrich_papers_semantically(openai_api_key=openai.api_key)

8. query database on the basis of a simple question
answer, sources = smart_expander.ask_question(
    query="What is the current clinical evidence for pegcetacoplan in treating GA?",
    top_k=10,
    openai_api_key=openai.api_key
)

print("🧠 GPT-4 Answer:\n", answer)

print("\n📎 Sources used:")
for paper in sources:
    print(f"- {paper['title']} (PMID: {paper['pmid']})")
