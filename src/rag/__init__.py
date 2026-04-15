"""KG-RAG question-answering package (Phase 3).

Two approaches share this package:

- ``text_to_cypher`` (Approach A, classical baseline): LLM generates a
  read-only Cypher query against the Phase 2 knowledge graph, the query is
  executed, and the results are formatted into an English answer with
  source-document citations.

- ``graph_rag`` (Approach B, main contribution): a document vector index
  over the corpus, combined with graph-expansion neighborhoods around
  retrieved docs, feeds both textual context and structured facts to the
  LLM for answering.

Both approaches implement the same ``BaseQASystem`` protocol so Phase 4
can benchmark them head-to-head with the same eval harness.
"""
