# Multiple-choice questions (MCQ) from Latvian Centralized High School Exams

## Data source

Data is gathered from the [National Centre for Education homepage](https://www.visc.gov.lv/lv/20222023-macibu-gada-uzdevumi#vidusskola)

## Prompt template

    {question}
    {answer_options}
    Atbildi formātā 'Atbilde ir X', kur X ir pareizās atbildes burts.

The answer is checked with case-insensitive regexp `Atbilde ir[*\s]*([A-Z])`

## Please cite
Roberts Darģis, Guntis Bārzdiņš, Inguna Skadiņa, Normunds Grūzītis, Baiba Saulīte. 2024. [Evaluating Open-Source LLMs in Low-Resource Languages: Insights from Latvian High School Exams].(https://aclanthology.org/2024.nlp4dh-1.28.pdf) 
Proceedings of the 4th International Conference on Natural Language Processing for Digital Humanities, Association for Computational Linguistics, 2024

@inproceedings{dargis-etal-2024-evaluating,
  author = "R. Dargis and G. Barzdins and I. Skadina and N. Gruzitis and B. Saulite",
  title = "Evaluating Open-Source LLMs in Low-Resource Languages: Insights from Latvian High School Exams",
  year = 2024,
  booktitle = "Proceedings of the 4th International Conference on Natural Language Processing for Digital Humanities",
  publisher = "Association for Computational Linguistics",
  pages = "289-293",
  month = "Nov",
  url = "https://aclanthology.org/2024.nlp4dh-1.28.pdf"
}
