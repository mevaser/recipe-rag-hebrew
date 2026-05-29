# Live Demo Plan

The goal of the live demo is to show a short, reliable example of the final recommended system without running expensive evaluation workflows.

## Recommended Demo Flow

1. Show the project folder structure:
   - `data/`
   - `src/`
   - `eval/`
   - `report/`
2. Show the final report:
   - `report/final_report.md`
3. Open the notebook:
   - `demo/live_demo_notebook.ipynb`
4. Run one safe local query from the notebook (the cell switches to project root first).
5. The demo script also creates `demo/demo_output.html` as a readable HTML backup file.
6. If terminal Hebrew rendering is hard to read, display `demo/demo_output.html` inside the notebook or open it in a browser.

The notebook uses `demo/run_demo_query.py`. The script uses the local backend by default, does not call OpenAI unless explicitly requested, does not run RAGAS, does not modify project data, and writes `demo/demo_output.html` as a presentation-friendly backup.

## Final Selected Live Question

`מה כמות הקמח, המים והשמרים בבצק הפיצה?`

Expected answer:

`320 גרם קמח 00 איטלקי, 188 גרם מים, 0.65 גרם שמרים יבשים (אינסטנט)`

## Exact Demo Command

```text
python demo\run_demo_query.py --question "מה כמות הקמח, המים והשמרים בבצק הפיצה?"
```

## Optional Backup Question

`כמה חזה עוף צריך למתכון חזה עוף וירקות בקרם קוקוס?`

## Backup Demo

If the live model call is unavailable during the presentation, use this backup flow:

1. Open `eval/final_evaluation_summary.md`
2. Open `report/final_report.md`
3. If available, open `demo/demo_output.html`
4. Show the retrieval metrics:
   - Hit@5 = 92%
   - Hit@20 = 94%
   - MRR = 0.8412
5. Show the local vs OpenAI frozen-context comparison:
   - OpenAI improved 8/12 rows and partially improved 2/12
6. Explain the final recommended candidate:
   - metadata-aware hybrid retrieval baseline
   - `strict_short_no_sources`
   - local qwen2.5 backend for offline demo
   - optional GPT-4.1-mini backend
   - rerank disabled by default
