# Final-test reporting incident

1. The one-time frozen final evaluation started normally after successful pre-flight and explicit authorization.
2. The official Track A test was accessed once.
3. T2 and T3 probabilities, EN2 probabilities, frozen binary predictions, and submission artifacts were persisted with SHA-256 hashes.
4. Final EN2 and T2 metrics were calculated and persisted in `final_test_results.json`.
5. The subsequent figure step failed exclusively because Matplotlib could not create `<MATPLOTLIB_CACHE>/fontlist-v330.json.matplotlib-lock`.
6. The pipeline stopped and recorded `FINAL_TEST_FAILED_AFTER_ACCESS`, as required by the frozen protocol.
7. No inference was repeated and no model or official test dataset was reopened during recovery.
8. Recovery was restricted to reporting from frozen outputs, with a project-local Matplotlib cache and the Agg backend.
9. No model, alpha, threshold, preprocessing rule, prediction, probability, or hyperparameter was changed.
