import json
from conllu import parse_incr

# jāņem lemma, nevis form, lai tiktu galā ar dīvainām krievu stila pēdiņām un tamlīdzīgi
# detaļās skaidrojumu ģenerēšanas loģika izskaidrota bakalaura darba tekstā, metodikas nodaļā

PUNCTUATION = {",", ".", "\"", ":", "!", "?", "(", ")"}
I_PUNCTUATION = {",", "\"", ":", "!", "?", "(", ")"} # visas apskatītās zīmes izņemot punkts, jo citādi tiek ņemti teikumi, piemēram, tikai ar vienu punktu
SUBORD_LABELS = {"ccomp", "csubj", "nsubj", "xcomp", "dep", "advcl", "root"}
SKIP_PUNCT = {"–", "-", ";"} # izlaižamās – tiks atmesti teikumi ar šādām pieturzīmēm

def extract_instruction_data(conllu_path, output_path, limit):
    dataset = []
    sentences = []

    with open(conllu_path, "r", encoding="utf-8") as file:
        for sentence in parse_incr(file):
            tokens = [token["lemma"] for token in sentence if token.get("lemma")]
            if any(token in I_PUNCTUATION for token in tokens) and not any(token in SKIP_PUNCT for token in tokens):
                sentences.append(sentence) # apstrādā tikai, ja satur kaut vienu no I_PUNCTUATION un nav sastopama neviena no izlaižamajām pieturzīmēm

    if limit: # var iestatīt ko grib, beigās tika paņemti 3000 no 6000 apstrādātiem teikumiem
        sentences = sentences[:limit]

    for sentence in sentences:
        quote_state = { # pēdiņu apstrādei
            "quotes": 0,
            "direct": False,
            "named_ent": False
        }
        punct_positions = [
            i for i, token in enumerate(sentence) if token.get("lemma") in PUNCTUATION #nosaka, kur ir pieturzīmes
        ]

        explanations = []
        for i, punct_idx in enumerate(punct_positions, 1): #katrai pieturzīmei
            punct = sentence[punct_idx]["lemma"]
            
            if punct == ",": # ja komats, tad jāatzīmē galva un atkarīgais elements
                clause_infos = explain_comma_usage(sentence, punct_idx)
                if clause_infos:
                    explanation = create_comma_explanation(sentence, clause_infos, punct_idx, i) # un veido skaidrojumu
                    explanations.append(explanation)
            else: #citādi vienkārši veido skaidrojumu
                explanation = explain_punctuation(sentence, punct_idx, i, quote_state)
                if explanation:
                    explanations.append(explanation)

        if explanations:
            marked_sentence = mark_all_punct_in_sentence(sentence) # atzīmē pieturzīmes formātā (i)
            instruction = "Balstoties uz latviešu valodas interpunkcijas un sintakses likumiem, īsi izskaidro katras pieturzīmes lietojumu dotajā tekstā."
            response = "\n".join(explanations)
            dataset.append({
                "instruction": instruction,
                "input": marked_sentence,
                "output": response
            })

    # saglabā jsonl failā
    with open(output_path, "w", encoding="utf-8") as out_f:
        for example in dataset:
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")

def explain_punctuation(sentence, punct_idx, i, quote_state): # zaurošanās f-ja visām pieturzīmēm, kas nav komats
    punct_token = sentence[punct_idx]
    punct_form = punct_token["lemma"]
    match punct_form:
        case ".":
            return explain_period_usage(i)
        case ":":
            return explain_colon_usage(sentence, punct_idx, i)
        case "\"":
            return explain_quote_usage(sentence, punct_idx, i, quote_state)
        case "(":
            return explain_parenthesis_usage(sentence, punct_idx, i)
        case ")":
            return explain_parenthesis_usage(i)
        case "!":
            return explain_exclamation_usage(sentence, punct_idx, i)
        case "?":
            return explain_question_usage(i)
        case _:
            return None

def explain_comma_usage(sentence, comma_idx): # fiksē komatam galvu un atkarīgo elementu
    comma_token = sentence[comma_idx]
    clauses = []
    token = find_token_by_id(sentence, comma_token["head"])
    head_id = token["head"]

    if head_id:
        head = find_token_by_id(sentence, head_id)
        clauses.append({
            "token": token,
            "head": head,
        })

    return clauses if clauses else None

def explain_period_usage(i): #punkts 
    return f"({i}) Punkts norāda teikuma beigas."

def explain_exclamation_usage(sentence, period_idx, i): # izsaukuma zīme
    prev_token = sentence[period_idx - 1]["upos"] if period_idx > 0 else ""
    if prev_token == "INTJ" or prev_token == "PART":
        return f"({i}) Izsaukuma zīme norāda izsaukuma vārdu."
    else:
        return f"({i}) Izsaukuma zīme norāda izsaukuma teikumu."

def explain_question_usage(i): # jautājuma zīme
    return f"({i}) Jautājuma zīme norāda jautājuma teikumu."

def explain_colon_usage(sentence, period_idx, i): # kols
    next_token = sentence[period_idx + 1] if period_idx + 1 < len(sentence) else ""
    if (next_token != "" and next_token["lemma"] == "\"") or next_token == "":
        return f"({i}) Kols ievada tiešo runu."
    else:
        if next_token["upos"] == "SCONJ" or next_token["upos"] == "CCONJ":
            return f"({i}) Kols atdala vienlīdzīgu teikuma daļu savienojumu."
        else:
            return f"({i}) Kols atdala vienlīdzīgu teikuma locekļu uzskaitījumu."

def explain_quote_usage(sentence, period_idx, i, quote_state): # pēdiņas, loģika izskaidrota bakalaura darba tekstā
    prev_token = sentence[period_idx - 1]["lemma"] if period_idx > 0 else ""
    if quote_state["quotes"] == 0:
        quote_state["quotes"] = 1
        if prev_token == ":":
            quote_state["direct"] = True
            return f"({i}) Pēdiņas atdala tiešo runu."
        else:
            for idx in range(period_idx + 1, len(sentence)):
                token = sentence[idx]
                if token["lemma"] == "\"":
                    prev_punct = sentence[idx - 1]
                    if prev_punct["upos"] == "PUNCT":
                        quote_state["direct"] = True
                        return f"({i}) Pēdiņas atdala tiešo runu."
                    else:
                        quote_state["named_ent"] = True
                        return f"({i}) Pēdiņas atdala nosaukto entitāti."
                elif idx == len(sentence) - 1:
                    quote_state["direct"] = True
                    return f"({i}) Pēdiņas atdala tiešo runu."
    else:
        quote_state["quotes"] = 0
        if quote_state["direct"]:
            quote_state["direct"] = False
            return f"({i}) Pēdiņas atdala tiešo runu."
        elif quote_state["named_ent"]:
            quote_state["named_ent"] = False
            return f"({i}) Pēdiņas atdala nosaukto entitāti."
        else:
            return f"({i}) Pēdiņas atdala XXXERRORERRORXXX."

def explain_parenthesis_usage(i): # iekavas
    return f"({i}) Iekavas atdala iespraudumu."

def mark_all_punct_in_sentence(sentence): # atzīmē visas pieturzīmes teikumā formātā (i)
    output = []
    comma_count = 1

    for token in sentence:
        id = token.get("id")
        if isinstance(token.get("id"), int):
            form = token.get("form")
            lemma = token.get("lemma")
            if lemma in PUNCTUATION:
                form = f"{lemma}({comma_count})"
                comma_count += 1

            output.append(form)
            #print(f"'{form}'")

            # tiek galā ar atstarpēm pēc vārdiem, kuriem nākamā tekstvienība ir pieturzīme
            if not token.get("misc") or token["misc"].get("SpaceAfter") != "No":
                output.append(" ")

    return "".join(output).strip()

def create_comma_explanation(sentence, clause_infos, token_idx, comma_index): # komats, loģika izskaidrota bakalaura darba tekstā
    clause = clause_infos[0]  # iegūst datus par galvu un atkarīgo
    token = clause['token']
    #print(f"'{token["form"]}', head: '{head["form"]}'")
    next_token = sentence[token_idx+1] if token_idx+1 < len(sentence) else ""
    if next_token != "":
        if next_token["lemma"] == "\"":
            return f"({comma_index}) Komats norāda tiešās runas beigas."
        
    if token["deprel"] in SUBORD_LABELS:
        feats = token.get("feats")
        if feats and feats.get("VerbForm") == "Conv":
            return f"({comma_index}) Komats atdala divdabja teicienu."
        else:
            return f"({comma_index}) Komats atdala palīgteikumu."
    elif token["deprel"] == "acl":
        if token["upos"] == "VERB":
            return f"({comma_index}) Komats atdala palīgteikumu."
        else:
            return f"({comma_index}) Komats atdala savrupinājumu."
    elif token["deprel"] == "conj":
        if token["upos"] == "VERB":
            return f"({comma_index}) Komats atdala vienlīdzīgas teikuma daļas."
        else:
            return f"({comma_index}) Komats atdala vienlīdzīgus teikuma locekļus."
    elif token["deprel"] == "parataxis" or token["deprel"] == "discourse":
        return f"({comma_index}) Komats atdala iespraudumu."
    elif token["deprel"] == "appos" or token["deprel"] == "obl":
        return f"({comma_index}) Komats atdala savrupinājumu."
    elif token["deprel"] == "vocative":
        return f"({comma_index}) Komats atdala uzrunu."
    else:
        return f"({comma_index}) Komats atdala XXXERRORERRORXXX."

def find_token_by_id(sentence, target_id): # atrod tekstvienību pēc tās id teikumā
    for token in sentence:
        token_id = token.get("id")
        if token_id == target_id:
            return token
    return None

if __name__ == "__main__": # lai palaistu funkciju vieglāk uz citiem failiem un citiem ierobežojumiem
    conllu_path = "lv_lvtb-ud-train.conllu"
    output_path = "latvian_punctuation_instructions.jsonl"
    extract_instruction_data(conllu_path, output_path, limit=6000)


# - acl
# - advcl
# - advmod
# - amod
# - appos
# - aux
# - case
# - cc
# - ccomp
# - conj
# - csubj
# - csubj:pass
# - dep
# - discourse
# - dislocated
# - flat
# - flat:name
# - iobj
# - mark
# - nmod
# - nsubj
# - nsubj:pass
# - nummod
# - obj
# - obl
# - orphan
# - parataxis
# - root
# - vocative
# - xcomp