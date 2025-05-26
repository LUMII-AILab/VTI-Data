# bak-darba-dati
Bakalaura darbā izmantotie skripti, iegūtie dati, un datu analīze.<br>
<br>
Failu sadalījums:
- benchmark_outputs<br>
  Satur visas etalonuzdevuma (manuālās novērtēšanas) modeļu izvades pirms pielāgošanas.
- eval_outputs<br>
  Satur visas automātiskās novērtēšanas modeļu izvades pēc pielāgošanas. Izvades sakārtotas mapēs, kas marķētas ar apskatītās epohas kārtas numuru.
- graphs<br>
  Satur bakalaura darbā iekļautie *loss*, *eval_loss* un *grad_norm* datus, .csv formātā lejupielādētus no TensorBoard rīka.
- jsonl_files<br>
  Satur visus pētījumā lietotos .jsonl failus, to starp treniņdatu kopu (sadalīta train, dev un test failos, attiecīgi 2500, 300 un 200 ieraksti), kā arī etalonuzdevuma datu kopu ar instrukciju latviešu un angļu valodā. Svarīgākie faili: lv_punct_*.jsonl, gold_standard.jsonl un gold_standard_eng.jsonl. Etalonuzdevuma dati – 2024.gada pasaules diktāts latviešu valodā.
- logs<br>
  Satur žurnālfailus ar termināļa izvadi no pielāgošanas.
- scripts<br>
  Satur visus izmantotos skriptus. Svarīgākie faili: generate_explanations.py (UDLV-LVTB parsētājs un automātisku skaidrojumu ģenerētājs), run_benchmark_task.py un run_benchmark_task_tuned.py (manuālā novērtēšana pirms un pēc pielāgošanas), lora_fine_tuning.py (pielāgošanas skripts) un eval_script.py (automātiskā novērtēšana). Pārējie ir palīgskripti un nav nepieciešami iepriekš minēto skriptu darbībai – tie tika izmantoti tikai atvieglotam darbam ar failiem.
- tuned_benchmark_outputs<br>
  Satur visas etalonuzdevuma (manuālās novērtēšanas) modeļu izvades pēc pielāgošanas. Izvades sakārtotas mapēs, kas marķētas ar apskatītās epohas kārtas numuru.
- pielāgošanas_analīze.xlsx<br>
  Satur iegūto datu analīzi, aprēķinus, grafikus.

