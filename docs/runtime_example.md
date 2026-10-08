# Recorded runtime example

A live collection session on **24 September 2026** queried mpox in Sierra Leone for **1 January to 31 December 2025**. It used Tavily search, live webpage/file retrieval, browser fallback, OCR, and model-assisted extraction. The saved provider was `anthropic`; the saved model identifier was `claude-sonnet-5`. This identifier describes that historical run, not a requirement for a new run.

The recorded execution started at **14:16:47 UTC** and completed at **18:59:30 UTC**: **4 h 42 min 43 s** elapsed. These are the original collection timestamps, not the time spent rebuilding its report afterward. Human review was disabled.

The run used an adaptive operation budget with a soft source target of 50 and the following effective limits:

| Operation | Effective limit | Recorded usage |
| --- | ---: | ---: |
| Search requests | 88 | 75 |
| Search results received | 380 | 219 |
| Webpage and file HTTP requests | 1,000 | 1,000 |
| Source acquisition targets | 200 | 200 |
| Structured extraction calls | 2,400 | 2,400 |
| Browser visits | 20 | 20 |
| OCR calls | 100 | 9 |

Other model roles had a default limit of 30 calls per role. The source-identity limit was 16; source-credibility suggestions and source-critic calls each had a limit of 4. An extraction reserve of 120 was recorded within the extraction budget. Limits are operation counts, not currency amounts.

The recorded stopping reason was `budget_exhausted`: the HTTP, source-target, extraction, and browser allowances were fully used. A completed execution therefore did not imply complete coverage. This is one measured session with its own settings, not the duration of the README's smaller `--quick-test-mode` run. Source availability, retries, model response time, and budget choices affect elapsed time.

For your own session, open the final report's **Run information and settings** section to see its recorded start/end times, model, budget use, and full settings. See [Configuration](configuration.md#budgets) to change operation limits.
