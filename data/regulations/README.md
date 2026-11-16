# Official legal texts

| File | Act | Language | Source |
|---|---|---|---|
| `ai_act_2024_1689_en.pdf` | Regulation (EU) 2024/1689 (AI Act), OJ L of 12.7.2024 | EN | EUR-Lex, CELEX 32024R1689 |
| `gdpr_2016_679_fr.pdf` | Regulation (EU) 2016/679 (GDPR), consolidated version of 04.05.2016 | FR | EUR-Lex, CELEX 02016R0679 |

Reuse is authorised (Commission Decision 2011/833/EU). Only the versions published in the
Official Journal of the European Union are authentic.

The indexed AI Act text is the **original version**. The amendments made by Regulation (EU)
2026/1744 ("Digital Omnibus on AI", in force since 27/07/2026) are reflected in the
**metadata** (`enforcement_date`, `enforcement_note`) and in the deterministic calendar
(`src/compliance/timeline.py`). To index a consolidated version, replace the PDF and run
`python -m src.rag.preprocess_laws --index --recreate`.

The GDPR is indexed from the official **French** version. Any official language version can
be used instead by adapting `GDPR_SPEC` in `src/rag/preprocess_laws.py` (heading regexes).
