# Data Manifest

## Corpus Name

Hebrew Recipe Corpus

## Domain

Cooking, recipes, and Hebrew culinary documents.

## Source of Documents

Private local recipe files collected by the project owner. The corpus includes recipe documents stored under `data/raw` and was prepared for a course RAG assignment.

## Number of Documents

- Processed documents after deduplication: `610`
- Generated chunks: `875`
- Indexed chunks: `873`

## Approximate Pages and Tokens

- The corpus is composed of PDF and DOCX recipe documents.
- Exact raw page counts vary by file and are not tracked in one single manifest field today.
- The processed pipeline currently works at the chunk level:
  - `875` generated chunks
  - `873` indexed chunks after filtering very short chunks
- This is sufficient to represent a medium-sized local recipe QA corpus for the assignment.

## File Types

- `.pdf`
- `.docx`

## License and Permission

The raw corpus is not committed to the repository because it may contain copyrighted and/or private recipe documents. Use of the corpus is limited to local educational work by the project owner. Redistribution rights are not assumed.

## Why This Corpus Is Suitable for RAG

- The documents are text-rich and contain factual recipe knowledge.
- Many questions require finding specific ingredients, quantities, cooking times, and process steps.
- The corpus contains overlapping and partially duplicated documents, which makes retrieval quality meaningful to evaluate.
- Some recipe names appear in filenames or categories rather than body text, making it a good test case for metadata-aware retrieval.

## Question Types

The system is intended to answer questions such as:

- factual questions
- ingredient questions
- numerical questions
- process and temporal questions
- comparison questions
- negation or absence questions

## Privacy and Sensitive Data Note

The repository excludes `data/raw/` through `.gitignore`. Raw files are treated as private local data and are not required for code review or public submission. Generated indexes and processed artifacts are also ignored when they can be rebuilt locally.
