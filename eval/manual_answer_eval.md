# Manual Answer Evaluation

This file is for human inspection only. No automatic answer grading was applied.

## Run Config

- Gold path: `C:\Users\mevas\Desktop\pythonProject\AI_Course\RAG\eval\gold_set.jsonl`
- k: `5`
- candidate_k: `50`
- rrf_k: `30`
- dense_weight: `0.5`
- bm25_weight: `2.0`

## Latest Verified Labels

This summary reflects the latest manually verified answer quality after the generation context-selection fixes. The file content below may show a smaller subset from the most recent helper run, but these labels are the current reviewed status for the tracked questions.

- Question 1: `correct`
  Notes: Answer remains correct.
- Question 6: `partial`
  Notes: Safe fallback is preferable to hallucination, but the user still does not receive the requested answer.
- Question 7: `partial`
  Notes: Safe fallback is preferable to hallucination, but the user still does not receive the requested answer.
- Question 9: `correct`
  Notes: Correct after neighbor-preserving generation context selection restored the bread temperature chunk.
- Question 13: `correct`
  Notes: Correct after recipe-specific context filtering removed unrelated hamburger and bolognese context.
- Question 14: `correct`
  Notes: Correct after recipe-specific context filtering and the grounded אינגריי taste extraction fix.
- Question 20: `correct`
  Notes: Answer remains correct.
- Question 30: `correct`
  Notes: Answer remains correct.
- Question 40: `correct`
  Notes: Answer remains correct.
- Question 49: `correct`
  Notes: Correct after the generation context-selection fixes.

## Results

### Question 1

Manual label: `correct`
Notes: Answer remains correct.

Question: כמה כיכרות מתקבלות במתכון ללחם ללא גלוטן כפרי, ומה המשקל של כל כיכר?

Reference answer: המתכון מיועד ל-2 כיכרות לחם ללא גלוטן כפרי, במשקל 915 גרם כל אחת.

Generated answer: תשובה: המתכון מיועד ל-2 כיכרות, במשקל 915 גרם כל אחת.
מקורות:
- docx:Recipes/2 ככרות לחם ללא גלוטן כפרי.docx_chunk_000

Expected chunk IDs:
- `docx:Recipes/2 ככרות לחם ללא גלוטן כפרי.docx_chunk_000`

Retrieved chunk IDs:
- `docx:Recipes/2 ככרות לחם ללא גלוטן כפרי.docx_chunk_000`
- `pdf:Recipes/לחם כללי + מחמצת/מדריך לאפיית לחמים.pdf:page-24_chunk_000`
- `pdf:Recipes/לחם כללי + מחמצת/מדריך לאפיית לחמים.pdf:page-43_chunk_001`
- `pdf:Recipes/לחם כללי + מחמצת/מדריך לאפיית לחמים.pdf:page-72_chunk_000`
- `docx:Recipes/2 ככרות לחם ללא גלוטן כפרי.docx_chunk_003`

Sources:
- `2 ככרות לחם ללא גלוטן כפרי.docx [docx:Recipes/2 ככרות לחם ללא גלוטן כפרי.docx_chunk_000]`
- `מדריך לאפיית לחמים.pdf [pdf:Recipes/לחם כללי + מחמצת/מדריך לאפיית לחמים.pdf:page-24_chunk_000]`
- `מדריך לאפיית לחמים.pdf [pdf:Recipes/לחם כללי + מחמצת/מדריך לאפיית לחמים.pdf:page-43_chunk_001]`
- `מדריך לאפיית לחמים.pdf [pdf:Recipes/לחם כללי + מחמצת/מדריך לאפיית לחמים.pdf:page-72_chunk_000]`
- `2 ככרות לחם ללא גלוטן כפרי.docx [docx:Recipes/2 ככרות לחם ללא גלוטן כפרי.docx_chunk_003]`

### Question 20

Manual label: `correct`
Notes: Answer remains correct.

Question: כמה חזה עוף צריך למתכון חזה עוף וירקות בקרם קוקוס?

Reference answer: צריך 500 גרם חזה עוף.

Generated answer: תשובה: צריך 500 גרם חזה עוף.
מקורות:
- pdf:Recipes/חזה עוף וירקות בקרם קוקוס - sweetdooly - סוויט דולי.pdf:page-1_chunk_000

Expected chunk IDs:
- `pdf:Recipes/חזה עוף וירקות בקרם קוקוס - sweetdooly - סוויט דולי.pdf:page-1_chunk_000`

Retrieved chunk IDs:
- `pdf:Recipes/חזה עוף וירקות בקרם קוקוס - sweetdooly - סוויט דולי.pdf:page-1_chunk_000`
- `pdf:Recipes/חזה עוף וירקות בקרם קוקוס - sweetdooly - סוויט דולי.pdf:page-2_chunk_000`
- `docx:Recipes/קציצות עוף יאמי.docx_chunk_000`
- `docx:Recipes/שיפודי נקניקיות הפתעה.docx_chunk_000`
- `docx:Recipes/עור עוף ממולא-החמין המפואר מהמטבח העיראקי.docx_chunk_000`

Sources:
- `חזה עוף וירקות בקרם קוקוס - sweetdooly - סוויט דולי.pdf [pdf:Recipes/חזה עוף וירקות בקרם קוקוס - sweetdooly - סוויט דולי.pdf:page-1_chunk_000]`
- `חזה עוף וירקות בקרם קוקוס - sweetdooly - סוויט דולי.pdf [pdf:Recipes/חזה עוף וירקות בקרם קוקוס - sweetdooly - סוויט דולי.pdf:page-2_chunk_000]`
- `קציצות עוף יאמי.docx [docx:Recipes/קציצות עוף יאמי.docx_chunk_000]`
- `שיפודי נקניקיות הפתעה.docx [docx:Recipes/שיפודי נקניקיות הפתעה.docx_chunk_000]`
- `עור עוף ממולא-החמין המפואר מהמטבח העיראקי.docx [docx:Recipes/עור עוף ממולא-החמין המפואר מהמטבח העיראקי.docx_chunk_000]`

### Question 30

Manual label: `correct`
Notes: Answer remains correct.

Question: מה חשוב לעשות לפני הכנסת הסלמון למחבת כדי שהעור לא יידבק?

Reference answer: חשוב שהמחבת תהיה חמה לפני שמכניסים את הדג, אחרת העור עלול להידבק למחבת.

Generated answer: תשובה: חשוב שהמחבת תהיה חמה לפני שמכניסים את הדג, אחרת העור יידבק למחבת.
מקורות:
- docx:Recipes/דג סלמון בחמאה במחבת.docx_chunk_000

Expected chunk IDs:
- `docx:Recipes/דג סלמון בחמאה במחבת.docx_chunk_000`

Retrieved chunk IDs:
- `docx:Recipes/דגים/סו ויד סלמון.docx_chunk_001`
- `pdf:Recipes/לחם כללי + מחמצת/מדריך לאפיית לחמים.pdf:page-51_chunk_001`
- `docx:Recipes/לחם כללי + מחמצת/מה זה מחמצת.docx_chunk_000`
- `docx:Recipes/דג סלמון בחמאה במחבת.docx_chunk_000`
- `docx:Recipes/דגים/סו ויד סלמון.docx_chunk_000`

Sources:
- `סו ויד סלמון.docx [docx:Recipes/דגים/סו ויד סלמון.docx_chunk_001]`
- `מדריך לאפיית לחמים.pdf [pdf:Recipes/לחם כללי + מחמצת/מדריך לאפיית לחמים.pdf:page-51_chunk_001]`
- `מה זה מחמצת.docx [docx:Recipes/לחם כללי + מחמצת/מה זה מחמצת.docx_chunk_000]`
- `דג סלמון בחמאה במחבת.docx [docx:Recipes/דג סלמון בחמאה במחבת.docx_chunk_000]`
- `סו ויד סלמון.docx [docx:Recipes/דגים/סו ויד סלמון.docx_chunk_000]`

### Question 40

Manual label: `correct`
Notes: Answer remains correct.

Question: מה הטעות הכי גדולה שאפשר לעשות בהכנת חומוס ביתי?

Reference answer: הטעות הכי גדולה היא להתחיל לטחון את הגרגרים כשהם לא מספיק רכים; אם נשאר קמצוץ קשה צריך להמשיך לבשל.

Generated answer: תשובה: הטעות הכי גדולה היא להתחיל לטחון את הגרגרים כשהם עדיין לא מספיק רכים; אם נשאר קמצוץ קשה, צריך להמשיך לבשל.
מקורות:
- pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-2_chunk_000

Expected chunk IDs:
- `pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-2_chunk_000`

Retrieved chunk IDs:
- `pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-2_chunk_000`
- `pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-2_chunk_001`
- `pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-4_chunk_001`
- `docx:Recipes/חומוס/מתכון לחומוס ביתי מושלם.docx_chunk_000`
- `pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-1_chunk_001`

Sources:
- `איך להכין חומוס ביתי מושלם.pdf [pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-2_chunk_000]`
- `איך להכין חומוס ביתי מושלם.pdf [pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-2_chunk_001]`
- `איך להכין חומוס ביתי מושלם.pdf [pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-4_chunk_001]`
- `מתכון לחומוס ביתי מושלם.docx [docx:Recipes/חומוס/מתכון לחומוס ביתי מושלם.docx_chunk_000]`
- `איך להכין חומוס ביתי מושלם.pdf [pdf:Recipes/חומוס/איך להכין חומוס ביתי מושלם.pdf:page-1_chunk_001]`
