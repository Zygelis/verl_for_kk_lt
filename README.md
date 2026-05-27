Papildyta https://github.com/verl-project/verl kodo bazė bakalaurinio darbo tyrimui atlikti. 


Qwen3-1.7B modelio treniravimas taikant GRPO, DAPO ir Reinforce++ sustiprinimo mokymo metodus.


# Struktūra

Visas tyrimo metu sukurtas kodas yra aplanke `kk_lithuanian/`. Tai yra mokymo skriptai, atlygio funkcija, duomenų paruošimas, testavimo skriptai ir pagalbiniai įrankiai.

Testavimo rezultatai AIME, AMC, GSM8K duomenų rinkiniams pateikti `kk_lithuanian/math_eval/math_eval/eval_results/`.

Testavimo rezultatai „Riterių ir melagių“ testinių duomenų aibei pateikti `kk_lithuanian/kk_eval/`.

Atlygio funkcija yra faile `kk_lithuanian/kk_lt_reward_function.py`.

Treniravimo įrašai (log's) kiekvienam metodui `kk_lithuanian/logs/`.

Treniravimo proceso grafikai `kk_lithuanian/logs/training_graphs/`.


## Duomenų paruošimo scriptas

Duomenų paruošimo skriptas yra faile `kk_lithuanian/data_preprocessing/kk_lithuanian.py`. Skriptas konvertuoja `JSONL` formato neapdorotus duomenis į `.parquet` formatą, tinkamą mokymui „Verl“ kodo bazėje.

**Paleidimas su numatytaisiais parametrais (sunkūs uždaviniai, Qwen3):**
```bash
python kk_lithuanian/data_preprocessing/kk_lithuanian.py
```

**Paleidimas su pasirinktais parametrais:**
```bash
python kk_lithuanian/data_preprocessing/kk_lithuanian.py \
  --data_path kk_lithuanian/raw_data/kk_full/kk_lt_train_easy.jsonl \
  --local_dir kk_lithuanian/data/qwen3_full/easy \
  --train_size 1000
```

**Pagrindiniai parametrai:**
- `--data_path`: sugeneruotų „Riterių ir melagių“ duomenų rinkinio failas (JSONL failas)
- `--local_dir`: išvesties direktorija (parquet failai)
- `--train_size`: treniravimo duomenų kiekis
- `--val_size`: testavimo arba validavimo duomenų kiekis




## Treniravimo paleidimas

**GRPO** metodas:
```bash
bash kk_lithuanian/training_scripts/grpo/grpo_qwen_3_1.7B_3072.sh
```

**DAPO** metodas:
```bash
bash kk_lithuanian/training_scripts/dapo/dapo_qwen3_1.7B_3072.sh
```

**Reinforce++** metodas:
```bash
bash kk_lithuanian/training_scripts/reinforce_plus_plus/re_plus_plus_qwen_3_1.7B_3072.sh
```

## Testavimo skriptai

Modelių testavimui naudoti skriptai:
- `kk_lithuanian/eval_lora_model.py` - modelio testavimas „Riterių ir melagių“ testinių duomenų aibei.
- `kk_lithuanian/math_eval/test_aime_amc_gsm8k.py` - modelio testavimas AIME, AMC, GSM8K duomenų rinkiniams.

- `kk_lithuanian/run_eval_lora_models.py` - `kk_lithuanian/eval_lora_model.py` paleidimas keliems modeliams.
- `kk_lithuanian/math_eval/run_eval_batch.py` - `kk_lithuanian/math_eval/test_aime_amc_gsm8k.py` paleidimas keliems modeliams.
