# Natural Language Generation (NLG) from Latvian Centralized High School Exams

## Data source

Data is gathered from the [National Centre for Education homepage](https://www.visc.gov.lv/lv/20222023-macibu-gada-uzdevumi#vidusskola)

## Open-source LLM models tested

    gemma2:27b-instruct-fp16
    llama3.1:405b-instruct_q5
    mistral-large:123b-instruct-2407-fp16
    qwen2-72b-instruct-fp16
    gpt-4o (August 19, 2024)
    

Tested in Ollama [https://ollama.com/] environment with default parameters, except: ContextLength=16384 and MaxTokens(num_predict)=2048

