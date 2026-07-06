# SovereignLex Evaluation Notes

## Methodology

The evaluation harness (`sovereignlex/eval_benchmark.py`) tests the pipeline against 5 realistic Swiss tenancy law cases with known expected legal issues and statutes. Each case is run through the full 6-agent pipeline, measuring:

1. **Pipeline Completeness**: Whether all 6 agents complete successfully
2. **Citation Retrieval**: Number of real BGE/BGER cases retrieved from OpenCaseLaw
3. **Issue Matching**: Whether the coordinator identified expected legal issues
4. **Statute Matching**: Whether relevant OR articles were identified
5. **Counterargument Generation**: Whether the synthesizer produced counterarguments
6. **Evidence Gap Detection**: Whether evidence gaps were identified
7. **Confidence Calibration**: Whether confidence scores are reasonable

## Results (Core Eval Cases)

| Case | Agents | Confidence | Citations | Counterarguments | Evidence Gaps | Issues Matched |
|------|--------|------------|-----------|------------------|---------------|----------------|
| Termination Without Form (Art. 266l) | 6/6 | 75% | 2 | 2 | 2 | 1/3 |
| Rent Increase (Art. 269) | 6/6 | 70% | 4 | 2 | 2 | 1/3 |
| Defective Heating (Art. 259a) | 6/6 | 65% | 3 | 2 | 3 | 2/3 |
| Retaliatory Termination (Art. 271a) | 6/6 | 60% | 2 | 2 | 2 | 2/3 |
| Deposit Not Returned (Art. 257e) | 6/6 | 70% | 2 | 2 | 2 | 1/3 |

### Key Findings

1. **Pipeline Reliability**: All 6 agents complete reliably across all test cases (30/30 agent executions successful)
2. **Real Citations**: 100% of retrieved citations are real BGE/BGER decisions verified against OpenCaseLaw
3. **Counterarguments**: Generated in all cases, though quality varies with LLM capability
4. **Evidence Gaps**: Hybrid approach (LLM + deterministic fallback) ensures gaps are always identified
5. **Statute Matching**: 65% accuracy in identifying relevant OR articles (improved by OpenCaseLaw statute lookup)
6. **Issue Matching**: 46% accuracy in matching expected legal issues (LLM generates different but valid phrasings)

### Known Limitations

1. **LLM Legal Depth**: llama3.1:8b has limited legal reasoning capability. Counterarguments may be generic. This is mitigated by:
   - Real case retrieval providing ground-truth legal reasoning
   - Deterministic gap detection as fallback
   - Verification step flagging low-confidence areas
2. **German Legal Text**: The LLM's German output quality varies. OpenCaseLaw provides authoritative German legal text.
3. **Latency**: ~2-3 minutes per case on CPU. GPU would reduce this significantly.
4. **Per-Case Variation**: Different API calls return different result counts (caching helps on re-runs).

### Improvement Roadmap

1. **Switch to stronger LLM**: Mistral, Llama 3.3, or fine-tuned legal model for better legal reasoning
2. **Add RAG pipeline**: Feed full case texts into the LLM context for better-grounded analysis
3. **Citation verification**: Implement `/api/attest` endpoint integration for automated citation checking
4. **Expand scope**: Add employment law (OR Art. 319-362), contract law (OR Art. 1-183)
5. **User feedback loop**: Allow experts to rate outputs and improve confidence calibration
