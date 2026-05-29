# Live Demo Notebook Notes

## Open the Notebook in VS Code
1. Open the project folder in VS Code.
2. Open `demo/live_demo_notebook.ipynb`.
3. Select a Python Jupyter kernel if VS Code asks for one.

## Run the Demo Cell
1. Go to `## 3. Run the Demo`.
2. Run the code cell.
3. The cell switches to project root first, so it still works if Jupyter opens from inside `demo/`.
4. The cell calls `demo/run_demo_query.py` with the selected live question:
   `מה כמות הקמח, המים והשמרים בבצק הפיצה?`

## If Hebrew Output Is Hard to Read
1. Go to `## 4. Open HTML Output`.
2. First run the demo cell, because `demo/demo_output.html` is created by `demo/run_demo_query.py`.
3. Run the HTML display cell.
4. If `demo/demo_output.html` exists, it will be displayed inside the notebook.
5. If needed, open `demo/demo_output.html` separately in a browser.

## Exact CLI Command (Same Question)

```text
python demo\run_demo_query.py --question "מה כמות הקמח, המים והשמרים בבצק הפיצה?"
```

## Recommended Backup Question

`כמה חזה עוף צריך למתכון חזה עוף וירקות בקרם קוקוס?`
