# Missing Context Investigation

## Goal
Investigate rows where retrieval failed to provide the expected context.

## Questions Investigated
- Q4
- Q5
- Q10
- Q35
- Q46

## Per-Question Findings
### Q4
- question: למה משרים את תערובת הגרעינים לפני שמפזרים אותה על הלחם?
- expected/reference answer: משרים את תערובת הגרעינים במים כדי שלא תישרף בזמן האפייה, בגלל שהיא נמצאת בחלק העליון של הלחם.
- expected chunk exists in processed chunks: yes
- appears in dense top results: yes
- appears in BM25 top results: yes
- appears in hybrid top results: yes
- expected chunk rank in dense: 22
- expected chunk rank in BM25: 2
- expected chunk rank in hybrid: 1
- suspected reason: expected_chunk_exists_but_ranked_low
- recommended fix: add_source_title_boosting

### Q5
- question: באיזו מהירות וכמה זמן לשים את בצק הלחם ללא גלוטן אחרי שמתקבל בצק אחיד יחסית?
- expected/reference answer: אחרי שמתקבל בצק אחיד יחסית, לשים במהירות בינונית כ-10 עד 15 דקות, עד שמתקבל מרקם סיבי ותפוס סביב וו הלישה.
- expected chunk exists in processed chunks: yes
- appears in dense top results: yes
- appears in BM25 top results: yes
- appears in hybrid top results: yes
- expected chunk rank in dense: 37
- expected chunk rank in BM25: 1
- expected chunk rank in hybrid: 4
- suspected reason: expected_chunk_exists_but_ranked_low
- recommended fix: add_source_title_boosting

### Q10
- question: כמה זמן מומלץ לא לפתוח את הלחם לאחר הוצאתו מהתנור?
- expected/reference answer: לא לפתוח את הלחם לפחות חצי שעה מרגע ההוצאה, ומומלץ להמתין לפחות שעה.
- expected chunk exists in processed chunks: yes
- appears in dense top results: yes
- appears in BM25 top results: yes
- appears in hybrid top results: yes
- expected chunk rank in dense: 16
- expected chunk rank in BM25: 1
- expected chunk rank in hybrid: 2
- suspected reason: expected_chunk_exists_but_ranked_low
- recommended fix: add_source_title_boosting

### Q35
- question: כמה זמן ובאיזו טמפרטורה אופים את ראשי השום לחומוס ללא גלוטן?
- expected/reference answer: עוטפים 6 ראשי שום בנייר אפייה ונייר כסף ואופים בתנור חם על 200 מעלות למשך 50 דקות.
- expected chunk exists in processed chunks: yes
- appears in dense top results: yes
- appears in BM25 top results: yes
- appears in hybrid top results: yes
- expected chunk rank in dense: 9
- expected chunk rank in BM25: 1
- expected chunk rank in hybrid: 1
- suspected reason: expected_chunk_exists_but_ranked_low
- recommended fix: add_source_title_boosting

### Q46
- question: כמה זמן מבשלים קובה בורגול מבושלת לאחר שמכניסים אותה למים הרותחים?
- expected/reference answer: מבשלים על להבה נמוכה כ-30 דקות או עד שהקובה מוכנה.
- expected chunk exists in processed chunks: yes
- appears in dense top results: yes
- appears in BM25 top results: yes
- appears in hybrid top results: yes
- expected chunk rank in dense: 6
- expected chunk rank in BM25: 1
- expected chunk rank in hybrid: 2
- suspected reason: expected_chunk_exists_but_ranked_low
- recommended fix: add_source_title_boosting

## Summary
- expected_chunk_exists_but_ranked_low: 5

## Next Retrieval Fix Candidates
1. source/title/recipe-name boosting
2. query expansion
3. metadata extraction improvements
4. Hebrew/RTL text normalization
5. raw document extraction inspection
