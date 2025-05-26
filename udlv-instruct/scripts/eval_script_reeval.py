import json
from pathlib import Path

def normalize_line(s): # noņem eos tekstvienību
    return s.strip().replace("<|endoftext|>", "").strip()

def lines_match(ref, pred): # atgriež true, ja apskatītā rinda satur sagaidāmo skaidrojumu
    ref_norm = normalize_line(ref)
    pred_norm = normalize_line(pred)
    return ref_norm in pred_norm

def reevaluate(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_id = data.get("model_id", "unknown")
    examples = data.get("examples", [])

    total_correct_lines = 0
    total_lines = 0
    exact_match_count = 0
    outputs = []
    mistakes = []

    for ex in examples:
        ref_lines = ex["reference"].strip().splitlines()
        pred_lines = ex["generated"].strip().splitlines()

        # ignorē pirmo rindu halucināciju dēļ
        if pred_lines and not pred_lines[0].strip().startswith("(1)"):
            pred_lines = pred_lines[1:]

        # pārbauda, vai sagaidāmais skaidrojums ir apskatītajā rindā
        line_matches = [lines_match(r, p) for r, p in zip(ref_lines, pred_lines)]
        correct = sum(line_matches)
        total = len(ref_lines)

        total_correct_lines += correct
        total_lines += total

        is_exact_match = (
            len(ref_lines) == len(pred_lines)
            and all(lines_match(r, p) for r, p in zip(ref_lines, pred_lines))
        )
        if is_exact_match:
            exact_match_count += 1

        example_out = {
            "instruction": ex["instruction"],
            "input": ex["input"],
            "reference": ex["reference"],
            "generated": ex["generated"]
        }

        if not is_exact_match:
            wrong_lines = []
            for i, (r, p) in enumerate(zip(ref_lines, pred_lines)):
                if not lines_match(r, p):
                    wrong_lines.append({
                        "line_num": i + 1,
                        "expected": r.strip(),
                        "got": p.strip()
                    })
            example_out["mistake_lines"] = wrong_lines
            mistakes.append(example_out)

        outputs.append(example_out)

    # metrikas
    exact_match = exact_match_count / len(examples) # precīza sakritība
    line_accuracy = total_correct_lines / total_lines if total_lines else 0
    total_mistakes = total_lines - total_correct_lines

   # saglabā visu json failā
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "exact_match": exact_match,
            "exact_match_percent": f"{exact_match * 100:.2f}%",
            "global_line_accuracy": line_accuracy,
            "line_accuracy_percent": f"{line_accuracy * 100:.2f}%",
            "total_examples": len(examples),
            "total_example_mistakes": len(mistakes),
            "total_line_mistakes": f"{total_mistakes} / {total_lines}",
            "examples": outputs
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # reevaluate("eval_outputs2/eval_deepseek.json", "eval_outputs2/eval_deepseek_reeval.json")

    folders = ["eval_outputs2", "eval_outputs3", "eval_outputs4", "eval_outputs5", "eval_outputs6", "eval_outputs7"]
    models = ["deepseek", "gemma", "llama", "mistral", "qwen"]
    for folder in folders:
        for model in models:
            json_path = f"eval_outputs/{folder}/eval_{model}.json"
            output_path = f"eval_outputs/{folder}/eval_{model}_reeval.json"
            reevaluate(json_path, output_path)
