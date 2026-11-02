# Insurance Claim Coverage Checker

Describe a home insurance claim and the app decides, from the policy wording alone, whether it is
**COVERED**, **NOT COVERED** or **UNCLEAR**. Every answer quotes the clauses it relies on, with page numbers.

The verdict matters less than the justification. When the policy is ambiguous, or the claim leaves out a
fact the policy depends on, the app answers UNCLEAR and says what information would settle it.
A substantiated "I don't know" is more useful than a confident answer without sources.

**Live demo:** [add your Streamlit link here]

## Test cases

The app is tested on a real, publicly published policy:
[RSA Home Contents Insurance policy wording (UK, 2025)](https://static.rsagroup.com/rsa/tenants-contents-insurance/NHC-wording.pdf).
The PDF is downloaded from the insurer's site on first run and is not stored in this repository.

| Claim | Verdict | Why |
|---|---|---|
| My bike was stolen from the building's communal storage area, which wasn't locked. | UNCLEAR | Optional bicycle cover (p. 46) only covers theft in a public place if the bike was locked to something fixed. The policy doesn't define "public place", and a shared storage room isn't clearly part of "home" as defined on p. 11, which decides whether the main theft cover (p. 25) applies. |
| A candle left on the garden table during a holiday melted in the heat and damaged the table. | UNCLEAR | Only accidental damage cover could apply, and it requires the optional Simple+ cover in the schedule (p. 42). "Contents in the open" (p. 28) doesn't cover heat damage. Whether melting counts as "sudden" accidental damage (p. 9) or gradual damage is debatable. |

## What I learned while testing

- The policy prints cover and exclusions in two columns. Plain text extraction mixed them, so I extract each table row as one clause.
- An early version applied an exclusion for credit cards (p. 40) to the bike claim. I now label every exclusion with the cover item it belongs to, and rewrite each claim into policy wording before searching so the right base cover is found.

## How it works

1. **Parsing that keeps clauses intact.** Like many policies, this one prints "What is covered" and
   "What is not covered" side by side. Plain text extraction reads across both columns and mixes cover
   with exclusions. The app extracts each table row as one clause, keeping its cover and its exclusions
   together, and treats only the text outside the tables as running prose.
2. **One index per policy.** Each policy has its own FAISS index (local `all-MiniLM-L6-v2` embeddings),
   so clauses from different contracts never mix in one answer.
3. **Several searches per claim:** the claim itself, the claim rewritten into policy wording by Gemini, then exclusions and conditions, then defined terms.
   The clause that decides a claim is often an exclusion or a definition far from the cover
   (for example "accidental damage" must be *sudden*, and much of the cover stops when the home is *unoccupied*).
4. **Structured verdict.** Gemini 2.5 Flash checks cover → conditions → exclusions → definitions and returns
   a verdict, reasoning, citations and missing information in a fixed schema.
5. **Citation check.** Every quote is compared with the retrieved policy text. A quote that is not found
   word for word is flagged in the interface.

## Run locally

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key (never commit it):

```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

```bash
python -m streamlit run app.py
```

You can also upload your own policy PDFs from the app.

## Files

- `rag.py`: downloading, parsing, indexing, retrieval and the structured verdict
- `app.py`: Streamlit interface

## Limits

- Some cover depends on options in the customer's policy schedule (for example accidental damage
  cover), which the app does not have. It says so when this matters.
- The policy is UK English. Scanned PDFs would need OCR first.
- There is no automated evaluation yet. The next step is a set of claims with expected verdicts, run on every change.

## Scaling to hundreds of contracts

Searching raw text works for a few policies. For hundreds, I would extract each policy once into a
structured table (cover, conditions, exclusions, limits, defined terms, page), query that table first to
find the applicable clauses, and use the raw text only to quote them. New contracts are then compared
field by field instead of re-reading whole documents.

## Author

Harisha Kommoju
