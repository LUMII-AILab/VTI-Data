# Multiple-choice questions (MCQ) from Latvian Centralized High School Exams

## Data source

Data is gathered from the [National Centre for Education homepage](https://www.visc.gov.lv/lv/20222023-macibu-gada-uzdevumi#vidusskola)

## Prompt template

    {question}
    {answer_options}
    Atbildi formātā 'Atbilde ir X', kur X ir pareizās atbildes burts.

The answer is checked with case-insensitive regexp `Atbilde ir[*\s]*([A-Z])`

